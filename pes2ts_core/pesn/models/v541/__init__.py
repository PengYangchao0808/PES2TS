"""V5.4.1 model package (WP2: ``V541Model`` + interfaces; ADR-0032).

Public surface (frozen for WP3/WP4)::

    from pes2ts_core.pesn.models.v541 import (
        V541Model, V541_SCHEMA_VERSION,
        NodeDeltaEmbedding, register_embedding, build_embedding, ...)
    model = V541Model()                          # E0: exact legacy parity
    model = V541Model(embedding_id="E1",
                      embedding_init_seed=1234)  # WP3: typed refusal until implemented

Scope discipline: this package only ADDS the V541 layer.  The legacy
``StarEGNNV3`` / heads / trainer / configs stay untouched and keep loading
old checkpoints bit-for-bit.
"""

from pes2ts_core.pesn.models.v541.embedding_registry import (
    EMBEDDING_SPECS,
    EmbeddingSpec,
    FROZEN_EMBEDDING_IDS,
    MAIN_MATRIX_EMBEDDING_IDS,
    RESERVED_CANDIDATE_IDS,
    build_embedding,
    embedding_spec,
    implemented_embedding_ids,
    is_embedding_implemented,
    is_embedding_known,
    known_embedding_ids,
    register_embedding,
    registered_test_ids,
    registry_restore,
    registry_snapshot,
)
from pes2ts_core.pesn.models.v541.interfaces import (
    ALPHA_PARAM_KEY,
    DIAGNOSTIC_KEYS_V541,
    EMBEDDING_ID_E0,
    HIDDEN_V541,
    MEASURED_PARAMS_V541_E0_JOINT,
    V541_SCHEMA_VERSION,
    W_OUT_PARAM_KEY,
    Z_DIM_V541,
    NodeDeltaEmbedding,
    V541EmbeddingNotImplementedError,
    V541EmbeddingUnknownError,
    V541InitError,
    V541InterfaceError,
    V541LegalityError,
    V541LoadError,
    count_scalar,
    detached_rms,
    detached_scalar,
    normalize_branch_output,
)
from pes2ts_core.pesn.models.v541.model import (
    LEGACY_STATE_PREFIXES,
    V541Model,
    measured_params_v541_e0,
)

__all__ = [
    "ALPHA_PARAM_KEY",
    "DIAGNOSTIC_KEYS_V541",
    "EMBEDDING_ID_E0",
    "EMBEDDING_SPECS",
    "EmbeddingSpec",
    "FROZEN_EMBEDDING_IDS",
    "HIDDEN_V541",
    "LEGACY_STATE_PREFIXES",
    "MAIN_MATRIX_EMBEDDING_IDS",
    "MEASURED_PARAMS_V541_E0_JOINT",
    "RESERVED_CANDIDATE_IDS",
    "V541_SCHEMA_VERSION",
    "W_OUT_PARAM_KEY",
    "Z_DIM_V541",
    "NodeDeltaEmbedding",
    "V541EmbeddingNotImplementedError",
    "V541EmbeddingUnknownError",
    "V541InitError",
    "V541InterfaceError",
    "V541LegalityError",
    "V541LoadError",
    "V541Model",
    "build_embedding",
    "count_scalar",
    "detached_rms",
    "detached_scalar",
    "embedding_spec",
    "implemented_embedding_ids",
    "is_embedding_implemented",
    "is_embedding_known",
    "known_embedding_ids",
    "measured_params_v541_e0",
    "normalize_branch_output",
    "register_embedding",
    "registered_test_ids",
    "registry_restore",
    "registry_snapshot",
]
