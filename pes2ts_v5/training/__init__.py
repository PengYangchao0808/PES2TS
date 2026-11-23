"""Training package: the single trainer, callbacks, and config-driven sweep."""

from pes2ts_v5.training._util import resolve_device, set_seed, weights_sha256
from pes2ts_v5.training.callbacks import (
    CALLBACK_REGISTRY,
    Callback,
    TrainerState,
    available_callbacks,
    build_callbacks,
)
from pes2ts_v5.training.sweep import build_dataset, load_config, run_sweep
from pes2ts_v5.training.trainer import TrainConfig, V5Trainer

__all__ = [
    "CALLBACK_REGISTRY",
    "Callback",
    "TrainConfig",
    "TrainerState",
    "V5Trainer",
    "available_callbacks",
    "build_callbacks",
    "build_dataset",
    "load_config",
    "resolve_device",
    "run_sweep",
    "set_seed",
    "weights_sha256",
]
