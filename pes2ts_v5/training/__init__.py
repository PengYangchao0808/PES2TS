"""Training package: unified trainer + config-driven sweep."""
from pes2ts_v5.training.trainer import TrainConfig, V5Trainer, set_seed, weights_sha256
from pes2ts_v5.training.sweep import build_dataset, load_config, run_sweep

__all__ = ["TrainConfig", "V5Trainer", "set_seed", "weights_sha256",
           "build_dataset", "load_config", "run_sweep"]
