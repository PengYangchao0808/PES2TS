"""Experiment configuration contract (Pydantic v2) — the only YAML schema.

``config/experiments/*.yaml`` -> :func:`load_experiment_config` ->
:class:`ExperimentConfig`; see :mod:`pes2ts_v5.configs.schema` for the contract
(runnable configs live under ``config/experiments/``, e.g.
``v56_baseline_gru.yaml``).
"""

from pes2ts_v5.configs.schema import (
    SCHEMA_NAME,
    SYNTHETIC_SPLIT_POLICIES,
    ConfigError,
    CoreConfig,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    HeadConfig,
    LossComponent,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    config_hash,
    config_to_dict,
    load_experiment_config,
)

__all__ = [
    "SCHEMA_NAME",
    "SYNTHETIC_SPLIT_POLICIES",
    "ConfigError",
    "CoreConfig",
    "DataConfig",
    "EvalConfig",
    "ExperimentConfig",
    "HeadConfig",
    "LossComponent",
    "LossConfig",
    "ModelConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "config_hash",
    "config_to_dict",
    "load_experiment_config",
]
