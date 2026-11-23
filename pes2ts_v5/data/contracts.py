"""Lazy bridge to the data-zone ``pes2ts_data.contracts`` StandardBatch contract.

The V5.6 engine must not duplicate the batch contract (it lives in
``pes2ts_data.contracts``).  This module imports it **lazily** so that the engine
still works when ``pes2ts_data`` is not installed yet:

* :func:`contracts_available` tells callers whether the real contract is present;
* :func:`standard_batch_view` translates a legacy V5 trajectory collate batch with
  ``to_standard``, attaches per-sample ``provenance_id`` tokens and validates it
  with ``validate_batch(..., require_core=False)`` — raising the contract's typed
  ``ContractViolation`` (``MISSING_FIELD`` / ``TRUTH_FIELD_FORBIDDEN`` / ...)
  instead of a bare ``KeyError`` deep inside a model.

Legacy -> standard rename is performed by ``pes2ts_data.contracts.to_standard``
(``process_label`` -> ``process_target`` etc.); this module only supplies the
provenance tokens and calls validation.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import torch

CODES: tuple[str, ...] = (
    "MISSING_FIELD", "BAD_DTYPE", "BAD_SHAPE", "BAD_VALUE", "BAD_INDEX",
    "BAD_BATCH_TYPE", "UNKNOWN_FIELD", "SHAPE_MISMATCH", "PROVENANCE_MISSING",
    "TRUTH_FIELD_FORBIDDEN",
)


def contracts_available() -> bool:
    """True when ``pes2ts_data.contracts`` can be imported."""
    try:
        import pes2ts_data.contracts  # noqa: F401
    except ImportError:
        return False
    return True


def contract_version() -> str:
    """Version of the imported batch contract (``""`` when unavailable)."""
    try:
        from pes2ts_data.contracts import CONTRACT_VERSION
    except ImportError:
        return ""
    return str(CONTRACT_VERSION)


def ContractViolationType() -> type[Exception]:
    """Return the contract's ``ContractViolation`` class (raises ImportError if absent)."""
    from pes2ts_data.contracts import ContractViolation
    return ContractViolation


def provenance_tokens(metas: Sequence[Mapping[str, Any]] | None,
                      batch_size: int) -> list[str]:
    """Per-sample provenance tokens from the dataset meta rows."""
    tokens: list[str] = []
    for index in range(int(batch_size)):
        meta = metas[index] if metas is not None and index < len(metas) else {}
        if not isinstance(meta, Mapping):
            meta = {}
        token = (meta.get("trajectory_id") or meta.get("traj_id")
                 or meta.get("reaction_id") or meta.get("traj_idx"))
        tokens.append(f"traj:{token}" if token is not None else f"traj:index-{index}")
    return tokens


def _batch_size(batch: Mapping[str, Any]) -> int:
    for key in ("frame_feats", "static_feats", "delta_s", "lengths"):
        value = batch.get(key)
        if torch.is_tensor(value) and value.dim() >= 1:
            return int(value.shape[0])
    return 0


def standard_batch_view(
    batch: Mapping[str, Any],
    metas: Sequence[Mapping[str, Any]] | None = None,
    *,
    required_fields: Iterable[str] | None = None,
    require_core: bool = False,
    validate: bool = True,
) -> dict[str, Any] | None:
    """Validate a legacy collate batch against the shared contract.

    Returns the standardised mapping (or ``None`` when ``pes2ts_data`` is not
    installed / ``validate=False``).  Raises the contract's
    ``ContractViolation`` on any violation — callers must not catch-and-ignore.
    """
    try:
        from pes2ts_data.contracts import to_standard, validate_batch
    except ImportError:
        return None
    standard = to_standard(batch)
    standard["provenance_id"] = provenance_tokens(metas, _batch_size(batch))
    if validate:
        validate_batch(standard, required=tuple(required_fields or ()),
                       require_core=require_core)
    return standard


def assert_batch_fields(batch: Mapping[str, Any], required: Iterable[str]) -> None:
    """Raise the typed ``MISSING_FIELD`` violation when ``required`` is unmet."""
    from pes2ts_data.contracts import assert_fields
    assert_fields(batch, tuple(required))


__all__ = [
    "CODES",
    "ContractViolationType",
    "assert_batch_fields",
    "contract_version",
    "contracts_available",
    "provenance_tokens",
    "standard_batch_view",
]
