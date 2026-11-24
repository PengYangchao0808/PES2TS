"""Batch-level validation engine for the 3-layer contract (v2).

Internal module (public surface: :mod:`pes2ts_data.contracts`).  Owns the typed
helper API (:func:`required_fields` / :func:`assert_fields`) and
:func:`validate_batch`, which accepts any of the three batch layers plus plain
mappings.  Field-level checks live in
:mod:`pes2ts_data._contract_fields`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields as dataclass_fields
from typing import Any, Iterable

import numpy as np

from ._contract_fields import (
    ContractViolation,
    check_value,
    coerce_array,
    forbid_truth_fields,
    is_truth_field,
    node_counts,
    provenance_token,
)
from ._contract_specs import (
    BASE_FIELDS,
    COMPAT_FIELDS,
    CONTRACT_REQUIRED_FIELDS,
    FIELD_SPECS,
    GRAPH_FIELDS,
    IDENTITY_FIELDS,
    SPLIT_TAGS,
)

__all__ = [
    "ContractViolation",
    "assert_fields",
    "is_truth_field",
    "required_fields",
    "validate_batch",
]


def _batch_items(batch: Any) -> list[tuple[str, Any]]:
    """Return ``(key, value)`` pairs of a batch object or plain mapping."""
    from .contracts import BaseBatch

    if isinstance(batch, BaseBatch):
        return [(field.name, getattr(batch, field.name))
                for field in dataclass_fields(batch)
                if getattr(batch, field.name) is not None]
    if isinstance(batch, Mapping):
        return list(batch.items())
    raise ContractViolation(
        "BAD_BATCH_TYPE",
        f"expected BaseBatch/SequenceBatch/GraphBatch or mapping, got "
        f"{type(batch).__name__!r}")


def _available_keys(batch: Any) -> set[str]:
    from .contracts import BaseBatch

    if isinstance(batch, BaseBatch):
        return {field.name for field in dataclass_fields(batch)
                if getattr(batch, field.name) is not None}
    if isinstance(batch, Mapping):
        return {str(key) for key in batch.keys() if isinstance(key, str)}
    raise ContractViolation(
        "BAD_BATCH_TYPE",
        f"expected a batch object or mapping, got {type(batch).__name__!r}")


def required_fields(required: Iterable[str], available: Iterable[str]) -> list[str]:
    """Return the fields from ``required`` absent from ``available`` (order kept)."""
    available_set = {str(name) for name in available}
    missing: list[str] = []
    seen: set[str] = set()
    for name in required:
        key = str(name)
        if key in available_set or key in seen:
            continue
        seen.add(key)
        missing.append(key)
    return missing


def assert_fields(batch: Any, required: Iterable[str]) -> None:
    """Raise ``MISSING_FIELD`` when a required field is absent from ``batch``."""
    for name in required:
        if is_truth_field(name):
            raise ContractViolation(
                "TRUTH_FIELD_FORBIDDEN",
                f"required field {name!r} looks like ground truth; models may "
                "not require truth fields as input (R1)", str(name))
    available = _available_keys(batch)
    missing = required_fields(required, available)
    if missing:
        raise ContractViolation(
            "MISSING_FIELD",
            f"missing required batch field(s): {', '.join(missing)}; "
            f"available: {', '.join(sorted(available)) or '<none>'}", missing[0])


def _layer_of(batch: Any, present: Mapping[str, Any]) -> str:
    from .contracts import BaseBatch, GraphBatch, SequenceBatch, StandardBatch

    if isinstance(batch, StandardBatch):
        return "legacy"
    if isinstance(batch, GraphBatch):
        return "graph"
    if isinstance(batch, SequenceBatch):
        return "sequence"
    if isinstance(batch, BaseBatch):
        return "base"
    if any(name in present for name in GRAPH_FIELDS):
        return "graph"
    if any(name in present for name in COMPAT_FIELDS):
        return "sequence"
    if any(name in present for name in BASE_FIELDS):
        return "base"
    return "graph"  # empty mapping -> surface the geometric MISSING_FIELD


def validate_batch(
    batch: Any,
    required: Iterable[str] = (),
    *,
    require_core: bool = True,
    require_identity: bool = True,
    require_provenance: bool = True,
    allow_unknown: bool = True,
) -> dict[str, Any]:
    """Validate a base/sequence/graph batch (or plain mapping) and normalize it.

    Parameters
    ----------
    batch:
        ``BaseBatch`` / ``SequenceBatch`` / ``GraphBatch`` (or the deprecated
        ``StandardBatch``) or a plain mapping with the same keys.
    required:
        Extra model-declared fields (``REQUIRED_FIELDS``) that must be present.
    require_core:
        Enforce ``pos``/``z``/``edge_index`` for graph-shaped batches.
    require_identity:
        Enforce ``sample_id`` + ``provenance`` (R1 identity); pass ``False`` for
        pure legacy trajectory batches built by the compat shim.
    require_provenance:
        Enforce that provenance-sensitive fields carry provenance.
    allow_unknown:
        When ``False``, keys outside ``FIELD_SPECS`` raise ``UNKNOWN_FIELD``.

    Returns
    -------
    dict
        Normalized report (``contract_version``, ``layer``, per-field metadata,
        identity values, compat/unknown field lists).
    """
    forbid_truth_fields(batch)
    present: dict[str, Any] = {}
    for key, value in _batch_items(batch):
        if not isinstance(key, str):
            raise ContractViolation("BAD_BATCH_TYPE",
                                    f"batch keys must be str, got "
                                    f"{type(key).__name__!r}")
        present[key] = value

    layer = _layer_of(batch, present)
    unknown = sorted(name for name in present if name not in FIELD_SPECS)
    if unknown and not allow_unknown:
        raise ContractViolation("UNKNOWN_FIELD",
                                f"unknown batch field(s): {', '.join(unknown)}; "
                                f"known: {', '.join(FIELD_SPECS)}", unknown[0])

    effective_required: list[str] = []
    graph_shaped = layer in ("graph", "legacy") or any(
        name in present for name in GRAPH_FIELDS)
    if require_core and graph_shaped:
        effective_required.extend(CONTRACT_REQUIRED_FIELDS)
    effective_required.extend(str(name) for name in required)
    if require_identity:
        effective_required.extend(IDENTITY_FIELDS)
    assert_fields(batch, effective_required)

    report_fields: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    for name, value in present.items():
        spec = FIELD_SPECS.get(name)
        if spec is None:
            continue  # tolerated legacy passthrough key
        report_fields[name] = check_value(spec, value)
        if spec.dtype in ("float", "int", "bool"):
            arrays[name] = coerce_array(value, name)

    if "sample_id" in present:
        sample_id = present["sample_id"]
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ContractViolation("BAD_VALUE", "sample_id must be a non-empty str",
                                    "sample_id")
    if "split_tag" in present:
        split_tag = present["split_tag"]
        if not isinstance(split_tag, str) or split_tag not in SPLIT_TAGS:
            raise ContractViolation(
                "SPLIT_TAG_INVALID",
                f"split_tag must be one of {SPLIT_TAGS}, got {split_tag!r}",
                "split_tag")

    token = None
    if "provenance" in present:
        token = provenance_token(present["provenance"])

    atoms = node_counts(arrays) if graph_shaped else None
    if atoms is not None:
        for name in ("edge_index", "sym_pair_idx"):
            if name not in arrays:
                continue
            index = arrays[name]
            if index.size and (index.min() < 0 or index.max() >= atoms):
                raise ContractViolation(
                    "BAD_INDEX",
                    f"indices out of range [0,{atoms}) (min={int(index.min())}, "
                    f"max={int(index.max())})", name)

    if require_provenance and token is None:
        sensitive = sorted(
            name for name in present
            if name in FIELD_SPECS and FIELD_SPECS[name].provenance_sensitive
            and name != "provenance")
        if sensitive:
            raise ContractViolation(
                "PROVENANCE_MISSING",
                "provenance-sensitive field(s) present without provenance: "
                f"{', '.join(sensitive)} (R1)", sensitive[0])

    return {
        "contract_version": "2.0.0",
        "layer": layer,
        "fields": report_fields,
        "required": effective_required,
        "sample_id": present.get("sample_id"),
        "split_tag": present.get("split_tag"),
        "targets": (sorted(present["targets"])
                    if isinstance(present.get("targets"), Mapping) else None),
        "provenance": token,
        "compat_fields": sorted(name for name in present if name in COMPAT_FIELDS),
        "unknown_fields": unknown,
        "deprecated": layer == "legacy",
    }
