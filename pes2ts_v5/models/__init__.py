"""Models: unified Encoder / memory Core / prediction Heads (V5.6)."""
from pes2ts_v5.models.encoders import (
    DescriptorEncoder,
    StarEGNNv2Encoder,
    build_encoder,
)
from pes2ts_v5.models.cores import (
    CORE_REGISTRY,
    CORE_META,
    CoreBase,
    build_core,
    count_params,
    resolve_core_config,
)
from pes2ts_v5.models.heads import MultiHead, compute_losses
from pes2ts_v5.models.v5_model import V5Model, build_v5_model

__all__ = [
    "DescriptorEncoder", "StarEGNNv2Encoder", "build_encoder",
    "CORE_REGISTRY", "CORE_META", "CoreBase", "build_core", "count_params",
    "resolve_core_config",
    "MultiHead", "compute_losses",
    "V5Model", "build_v5_model",
]
