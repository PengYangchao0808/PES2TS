"""Sweep-matrix contract: all 14 architectures must run through the ONE engine.

Regression guard for the gap where ``training/sweep.py`` still used the legacy
``build_v5_model`` path (encoder width vs resolved core width mismatch, 13/14
architectures crashing).  The sweep must assemble every architecture through
``models.registry.build_model_from_config`` with the dataset's real feature
dimensions, and report ``n_done == 14, n_failed == 0`` on the architecture-matrix
YAML with one batch per architecture.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from pes2ts_v5.configs.schema import config_to_dict, load_experiment_config
from pes2ts_v5.models.registry import list_cores
from pes2ts_v5.runner import resolve_dataset
from pes2ts_v5.training import sweep as sweep_mod

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "experiments"
ALL_CORES = list_cores()


def _matrix_doc(tmp_path: Path) -> dict:
    """Arch-matrix config with its synthetic fixture redirected into tmp_path."""
    doc = config_to_dict(load_experiment_config(CONFIG_DIR / "v56_arch_matrix.yaml"))
    doc["data"]["store_manifest"] = str(tmp_path / "synthetic_matrix")
    doc["training"]["epochs"] = 1
    doc["evaluation"]["cpu_benchmark"] = True
    return doc


def test_sweep_matrix_all_14_architectures_run(tmp_path, write_yaml) -> None:
    doc = _matrix_doc(tmp_path)
    config = write_yaml(doc, "matrix_sweep.yaml")

    summary = sweep_mod.run_sweep(
        config, archs=list(ALL_CORES), out_dir=tmp_path / "sweep",
        device="cpu", max_epochs=1, max_train_batches=1)

    assert summary["n_done"] == 14, summary["failures"]
    assert summary["n_failed"] == 0, summary["failures"]
    assert summary["archs"] == ALL_CORES
    assert summary["data_source"] == "synthetic_fixture"

    leaderboard = summary["leaderboard"]
    assert len(leaderboard) == 14
    seen = {row["arch"] for row in leaderboard}
    assert seen == set(ALL_CORES)
    for row in leaderboard:
        assert row["status"] == "done", row
        assert isinstance(row["core_params"], int) and row["core_params"] > 0
        assert isinstance(row["cpu_step_ms"], (int, float))
        assert row["cpu_step_ms"] > 0
        assert row["move_rmse"] is not None

    # registry CSV: exactly one row per architecture, all "done"
    registry_path = Path(summary["reports_dir"]) / "experiment_registry.csv"
    assert registry_path.exists()
    with registry_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 14
    assert [row["status"] for row in rows] == ["done"] * 14
    assert {row["arch"] for row in rows} == set(ALL_CORES)
    for row in rows:
        assert int(row["core_params"]) > 0
        assert float(row["cpu_step_ms"]) > 0

    # failure taxonomy exists (header) but has zero data rows when nothing failed
    taxonomy = Path(summary["reports_dir"]) / "failure_taxonomy.csv"
    assert taxonomy.exists()
    with taxonomy.open(encoding="utf-8-sig", newline="") as fh:
        assert list(csv.DictReader(fh)) == []

    # per-architecture artifacts exist and carry the shared identity fields
    run_dir = Path(summary["run_dir"])
    provenance = json.loads(
        (run_dir / "experiments" / "M04" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["weights_sha256"]
    assert provenance["config_hash"]
    assert (run_dir / "experiments" / "M04" / "checkpoints").is_dir()
    assert (run_dir / "dataset_manifest.json").exists()


def test_sweep_resolves_dataset_dims_from_data_not_constants(tmp_path, write_yaml) -> None:
    """Model assembly uses the dataset's declared dims (11/8 here), not magic."""
    doc = _matrix_doc(tmp_path)
    config = write_yaml(doc, "dims_sweep.yaml")
    cfg = load_experiment_config(config)
    dataset, info = resolve_dataset(cfg)
    assert info["dims"] == {"frame_feat_dim": 11, "static_feat_dim": 8, "q_dim": 3}
    assert dataset.frame_feat_dim == 11 and dataset.static_feat_dim == 8


def test_explicit_dim_override_must_match_dataset(tmp_path, write_yaml) -> None:
    from pes2ts_v5.data.resolve import DataDimensionMismatch

    doc = _matrix_doc(tmp_path)
    doc["data"]["frame_feat_dim"] = 7  # honest explicit override -> typed mismatch
    cfg = load_experiment_config(write_yaml(doc, "bad_dims.yaml"))
    with pytest.raises(DataDimensionMismatch, match="frame_feat_dim"):
        resolve_dataset(cfg)


def test_explicit_dim_override_accepted_when_correct(tmp_path, write_yaml) -> None:
    doc = _matrix_doc(tmp_path)
    doc["data"]["frame_feat_dim"] = 11
    doc["data"]["static_feat_dim"] = 8
    cfg = load_experiment_config(write_yaml(doc, "good_dims.yaml"))
    dataset, info = resolve_dataset(cfg)
    assert info["dims"]["frame_feat_dim"] == 11
    assert dataset.frame_feat_dim == 11


def test_variant_config_only_changes_the_core(tmp_path) -> None:
    cfg = load_experiment_config(CONFIG_DIR / "v56_arch_matrix.yaml")
    variant = sweep_mod._variant_config(cfg, "M14")
    assert variant.model.core.name == "M14"
    assert variant.model.encoder == cfg.model.encoder
    assert variant.model.head == cfg.model.head
    assert variant.loss == cfg.loss
    assert variant.training == cfg.training
    assert variant.data == cfg.data
    # every other model knob is inherited unchanged
    assert variant.model.core.hidden_dim == cfg.model.core.hidden_dim
    assert copy.deepcopy(cfg).model.core.name == cfg.model.core.name
