"""Encoder factory: descriptor (default) / StarEGNN v2 (optional stub)."""

from __future__ import annotations

from pes2ts_v5.models.encoders.descriptor import DescriptorEncoder
from pes2ts_v5.models.encoders.stregnn_v2 import StarEGNNv2Encoder

ENCODER_REGISTRY = {
    "descriptor": DescriptorEncoder,
    "stregnn_v2": StarEGNNv2Encoder,
}


def build_encoder(
    name: str,
    *,
    frame_feat_dim: int,
    static_feat_dim: int,
    hidden_dim: int,
    dropout: float = 0.0,
    **kwargs,
):
    key = (name or "descriptor").strip().lower()
    if key not in ENCODER_REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; choices={sorted(ENCODER_REGISTRY)}")
    cls = ENCODER_REGISTRY[key]
    if key == "descriptor":
        return cls(frame_feat_dim, static_feat_dim, hidden_dim, dropout=dropout)
    return cls(hidden_dim, dropout=dropout, **kwargs)
