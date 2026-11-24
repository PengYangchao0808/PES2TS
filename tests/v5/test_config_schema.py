"""Experiment-config contract: strict schema, hash stability, typed loader errors."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pes2ts_v5.configs.schema import (
    ConfigError,
    ExperimentConfig,
    config_hash,
    config_to_dict,
    load_experiment_config,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "experiments"

SHIPPED = (
    "smoke.yaml",
    "screening.yaml",
    "v56_baseline_gru.yaml",
    "v56_arch_matrix.yaml",
    "v56_egnn_sym.yaml",
    "v56_real_idx_v2.yaml",
)


@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_configs_load(name: str) -> None:
    cfg = load_experiment_config(CONFIG_DIR / name)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.experiment_id
    assert cfg.model.core.name
    assert cfg.loss.components


def test_smoke_config_contents(smoke_doc: dict) -> None:
    cfg = ExperimentConfig.model_validate(smoke_doc)
    assert cfg.experiment_id == "v56-smoke"
    assert cfg.seed == 17
    assert cfg.is_synthetic()
    assert cfg.model.encoder == "descriptor"
    assert cfg.model.core.name == "gru"
    assert cfg.model.core.hidden_dim == 32
    assert cfg.training.optimizer.name == "adamw"
    assert [c.name for c in cfg.loss.components] == [
        "move_huber", "process_ce", "progress_huber", "trust_bce"]
    assert cfg.training.callbacks


def test_config_hash_is_stable(smoke_doc: dict) -> None:
    first = config_hash(ExperimentConfig.model_validate(smoke_doc))
    second = config_hash(ExperimentConfig.model_validate(copy.deepcopy(smoke_doc)))
    assert first == second
    assert len(first) == 64
    int(first, 16)  # hex


@pytest.mark.parametrize("path,value", [
    (("seed",), 18),
    (("experiment_id",), "v56-smoke-2"),
    (("model", "core", "name"), "M06"),
    (("model", "core", "hidden_dim"), 64),
    (("training", "lr"), 0.002),
    (("training", "callbacks"), ["checkpoint_best"]),
    (("loss", "components"), [{"name": "move_huber", "weight": 2.0}]),
    (("evaluation", "splits"), ["id"]),
])
def test_config_hash_changes_when_a_field_changes(smoke_doc: dict,
                                                  path: tuple[str, ...],
                                                  value) -> None:
    base = config_hash(ExperimentConfig.model_validate(copy.deepcopy(smoke_doc)))
    doc = copy.deepcopy(smoke_doc)
    node = doc
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    changed = config_hash(ExperimentConfig.model_validate(doc))
    assert changed != base


def test_unknown_keys_rejected(smoke_doc: dict, write_yaml) -> None:
    for mutate in (
        lambda d: d.update(bogus=1),
        lambda d: d["data"].update(kind="synthetic"),
        lambda d: d["model"].update(hidden_dim=32),
        lambda d: d["model"]["core"].update(width=64),
        lambda d: d["training"].update(lr_schedule="cosine"),
        lambda d: d["evaluation"].update(rollout=True),
    ):
        doc = copy.deepcopy(smoke_doc)
        mutate(doc)
        with pytest.raises(ConfigError):
            load_experiment_config(write_yaml(doc))


def test_negative_loss_weight_rejected(smoke_doc: dict, write_yaml) -> None:
    doc = copy.deepcopy(smoke_doc)
    doc["loss"]["components"][0]["weight"] = -0.5
    with pytest.raises(ConfigError, match="weight"):
        load_experiment_config(write_yaml(doc))


def test_unknown_callback_rejected(smoke_doc: dict, write_yaml) -> None:
    doc = copy.deepcopy(smoke_doc)
    doc["training"]["callbacks"] = ["checkpoint_best", "not_a_callback"]
    with pytest.raises(ConfigError, match="callback"):
        load_experiment_config(write_yaml(doc))


def test_unknown_loss_component_rejected(smoke_doc: dict, write_yaml) -> None:
    doc = copy.deepcopy(smoke_doc)
    doc["loss"]["components"] = [{"name": "magic_loss", "weight": 1.0}]
    with pytest.raises(ConfigError, match="loss"):
        load_experiment_config(write_yaml(doc))


def test_physics_residual_is_declared_but_not_implemented(smoke_doc: dict,
                                                         write_yaml) -> None:
    from pes2ts_v5.losses.components import build_weighted_loss_from_config
    from pes2ts_v5.losses.registry import LossNotImplementedError

    doc = copy.deepcopy(smoke_doc)
    doc["loss"]["components"] = [
        {"name": "physics_residual", "weight": 0.1},
    ]
    cfg = load_experiment_config(write_yaml(doc))  # schema accepts (registered)
    loss = build_weighted_loss_from_config(cfg.loss)
    assert loss.describe()[0]["implemented"] is False
    from pes2ts_v5.models.registry import make_smoke_batch

    with pytest.raises(LossNotImplementedError) as excinfo:
        loss(make_smoke_batch(), make_smoke_batch())
    assert excinfo.value.code == "LOSS_NOT_IMPLEMENTED"


@pytest.mark.parametrize("hidden_dim", [16, 256, 1024, "auto"])
def test_hidden_dim_accepted(smoke_doc: dict, write_yaml, hidden_dim) -> None:
    doc = copy.deepcopy(smoke_doc)
    doc["model"]["core"]["hidden_dim"] = hidden_dim
    cfg = load_experiment_config(write_yaml(doc))
    assert cfg.model.core.hidden_dim == hidden_dim


@pytest.mark.parametrize("hidden_dim", [8, 15, 1025, 4096, "wide"])
def test_hidden_dim_rejected(smoke_doc: dict, write_yaml, hidden_dim) -> None:
    doc = copy.deepcopy(smoke_doc)
    doc["model"]["core"]["hidden_dim"] = hidden_dim
    with pytest.raises(ConfigError):
        load_experiment_config(write_yaml(doc))


@pytest.mark.parametrize("experiment_id", [
    "", "-bad", "has space", "has/slash", "x" * 65,
])
def test_experiment_id_pattern(smoke_doc: dict, write_yaml, experiment_id) -> None:
    doc = copy.deepcopy(smoke_doc)
    doc["experiment_id"] = experiment_id
    with pytest.raises(ConfigError):
        load_experiment_config(write_yaml(doc))


def test_loader_raises_config_error_for_bad_files(tmp_path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_experiment_config(tmp_path / "missing.yaml")
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("experiment_id: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_experiment_config(bad_yaml)
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("just-a-string\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_experiment_config(scalar)


def test_config_to_dict_is_strict_json(smoke_doc: dict) -> None:
    cfg = ExperimentConfig.model_validate(smoke_doc)
    doc = config_to_dict(cfg)
    text = json.dumps(doc, allow_nan=False)  # must not raise
    assert "NaN" not in text and "Infinity" not in text
    assert set(doc) == {"experiment_id", "seed", "data", "model", "loss",
                        "training", "evaluation"}
    # round-trip preserves the hash
    assert config_hash(ExperimentConfig.model_validate(doc)) == config_hash(cfg)


def test_callback_alias_accepted(smoke_doc: dict, write_yaml) -> None:
    doc = copy.deepcopy(smoke_doc)
    doc["training"]["callbacks"] = ["checkpoint_best", "provenance", "oom_guard"]
    cfg = load_experiment_config(write_yaml(doc))
    assert cfg.training.callbacks == ["checkpoint_best", "provenance", "oom_guard"]
