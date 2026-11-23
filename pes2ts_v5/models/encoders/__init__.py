"""Encoders: low-dimensional reaction descriptor (default) + StarEGNN v2 stub."""
from pes2ts_v5.models.encoders.descriptor import DescriptorEncoder
from pes2ts_v5.models.encoders.factory import build_encoder
from pes2ts_v5.models.encoders.stregnn_v2 import StarEGNNv2Encoder

__all__ = ["DescriptorEncoder", "StarEGNNv2Encoder", "build_encoder"]
