"""Feature normalization fitted on ``train_fit`` only (B3).

The v1 encoder consumes scale-free invariant bases (Gaussian distance
expansion, embeddings), so the normalizer's job is the low-dim descriptor
standardization used by the same-input MLP baseline and any pooled scalar
consumers.  It is fitted exclusively on the ``train_fit`` subset — sealed
subsets contribute zero samples to any fitted statistic — and serialized to
``normalization.json`` with its fitting scope bound by hashes.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from pes2ts_core.utils.hashing import JSONValue, stable_json_dumps
from pes2ts_core.pesn.features.graph import pooled_descriptors

NORMALIZATION_SCHEMA_VERSION: Final[str] = "pesn_normalization_v1"
DESCRIPTOR_NAMES: Final[tuple[str, ...]] = (
    "n_atoms",
    "n_heavy",
    "n_hydrogen",
    "n_driver_coords",
    "driver_mean",
    "driver_std",
    "driver_min",
    "driver_max",
    "r_bond_mean_distance",
    "p_bond_mean_distance",
    "driver_q_scaled_mean",
)


class NormalizationError(RuntimeError):
    """Typed failure of normalizer fit/apply."""


class DescriptorNormalizer:
    """Standardizes the pooled descriptor vector (mean/std per dimension)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        if mean.shape != std.shape or mean.ndim != 1:
            raise NormalizationError("mean/std must be equal-length 1-D arrays")
        self.mean = mean.astype(np.float64)
        self.std = np.where(std > 1e-12, std, 1.0).astype(np.float64)

    @classmethod
    def fit(cls, descriptors: np.ndarray) -> "DescriptorNormalizer":
        matrix = np.asarray(descriptors, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise NormalizationError(
                f"fit needs a non-empty 2-D descriptor matrix, got {matrix.shape}"
            )
        return cls(matrix.mean(axis=0), matrix.std(axis=0))

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        vector = np.asarray(descriptors, dtype=np.float64)
        if vector.shape != self.mean.shape:
            raise NormalizationError(
                f"descriptor shape {vector.shape} != fitted {self.mean.shape}"
            )
        return (vector - self.mean) / self.std

    def to_document(
        self, *, train_fit_dataset_sha256: str, n_samples: int
    ) -> dict[str, JSONValue]:
        return {
            "schema_version": NORMALIZATION_SCHEMA_VERSION,
            "descriptor_names": list(DESCRIPTOR_NAMES),
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "fit_scope": "train_fit_only",
            "train_fit_dataset_sha256": train_fit_dataset_sha256,
            "n_fit_samples": int(n_samples),
            "sealed_samples_contributed": 0,
        }

    @classmethod
    def from_document(cls, document: dict[str, JSONValue]) -> "DescriptorNormalizer":
        if document.get("schema_version") != NORMALIZATION_SCHEMA_VERSION:
            raise NormalizationError(
                f"normalization schema {document.get('schema_version')!r} unsupported"
            )
        if document.get("sealed_samples_contributed") != 0:
            raise NormalizationError("normalizer must not see sealed samples")
        return cls(
            np.asarray(document["mean"], dtype=np.float64),
            np.asarray(document["std"], dtype=np.float64),
        )


def descriptors_of(features) -> np.ndarray:
    """Pooled descriptors of one GraphFeatures (re-exported convenience)."""
    return pooled_descriptors(features)


def normalization_content_sha256(document: dict[str, JSONValue]) -> str:
    import hashlib

    return hashlib.sha256(
        stable_json_dumps(document).encode("utf-8")
    ).hexdigest()


__all__ = [
    "DESCRIPTOR_NAMES",
    "NORMALIZATION_SCHEMA_VERSION",
    "DescriptorNormalizer",
    "NormalizationError",
    "descriptors_of",
    "normalization_content_sha256",
]
