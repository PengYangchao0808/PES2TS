"""The single 3-layer data-batch contract for PES2TS (Zone 2, v2).

This module is the only place that declares the canonical batch field set,
per-field dtype/shape/provenance metadata and the typed rejection codes used by
every loader, trainer and evaluator.

Contract v2 (3 layers)
----------------------
``BaseBatch``
    Identity/provenance layer: ``sample_id``, ``targets`` (mapping),
    ``split_tag``, ``provenance``.  Only these four fields are validated at the
    base layer.
``SequenceBatch``
    Legacy V5.6 trajectory compat fields (``frame_feats``, ``static_feats``,
    ``lengths``, ``delta_s``, ``move_target``, ``process_target``,
    ``progress_target``, ``trust_target``), mapped 1:1 to the legacy trajectory
    dict.
``GraphBatch``
    Geometric batch fields (``pos``, ``z``, ``edge_index``, ``endpoint_mask``,
    ``sym_pair_idx``, ``target_xyz``, ``sample_weight``).
``StandardBatch``
    Deprecated v1 flat union of graph + sequence fields; it warns on
    construction and must only be used through the compat shim
    (:func:`to_standard` / :func:`from_standard`, also deprecated).

Constitution mapping
--------------------
R1  truth isolation: truth-like keys are rejected with
    ``TRUTH_FIELD_FORBIDDEN``; target/feature fields are
    ``provenance_sensitive`` and require ``provenance`` (``PROVENANCE_MISSING``).
R5  full-cost retention: ``sample_weight`` carries the per-sample cost weight.
R6  split firewall: ``split_tag`` is restricted to the frozen split vocabulary;
    the split guards live in :mod:`pes2ts_data.splits.policy`.
P2  typed honest rejection: every violation raises :class:`ContractViolation`
    with a stable ``.code`` -- never ``KeyError``/``TypeError``/silent
    imputation.

Dependencies: stdlib + :mod:`numpy` only.  Torch tensors are accepted through
duck typing (``.detach().cpu().numpy()``) without importing torch.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, Mapping

from ._contract_specs import (
    BASE_FIELDS,
    CODES,
    COMPAT_FIELDS,
    CONTRACT_REQUIRED_FIELDS,
    CONTRACT_VERSION,
    DTYPE_FAMILIES,
    FIELD_SPECS,
    GRAPH_FIELDS,
    IDENTITY_FIELDS,
    LEGACY_CONTRACT_VERSION,
    LEGACY_FIELD_MAP,
    LEGACY_TO_STANDARD,
    SPLIT_TAGS,
    STANDARD_TO_LEGACY,
    TRUTH_FIELD_PATTERNS,
    FieldSpec,
)
from ._contract_validation import (
    ContractViolation,
    assert_fields,
    is_truth_field,
    required_fields,
    validate_batch,
)

__all__ = [
    "BASE_FIELDS",
    "CODES",
    "COMPAT_FIELDS",
    "CONTRACT_REQUIRED_FIELDS",
    "CONTRACT_VERSION",
    "DTYPE_FAMILIES",
    "FIELD_SPECS",
    "GRAPH_FIELDS",
    "IDENTITY_FIELDS",
    "LEGACY_CONTRACT_VERSION",
    "LEGACY_FIELD_MAP",
    "LEGACY_TO_STANDARD",
    "SPLIT_TAGS",
    "STANDARD_TO_LEGACY",
    "TRUTH_FIELD_PATTERNS",
    "BaseBatch",
    "ContractViolation",
    "FieldSpec",
    "GraphBatch",
    "Provenance",
    "SequenceBatch",
    "StandardBatch",
    "assert_fields",
    "from_standard",
    "is_truth_field",
    "required_fields",
    "to_standard",
    "validate_batch",
]


@dataclass(frozen=True)
class Provenance:
    """Frozen provenance token for one batch sample (R1).

    Parameters
    ----------
    token:
        Stable non-empty provenance token (curve/split/artifact lineage).
    split_tag:
        Optional split tag carried by the provenance record itself.
    sha256:
        Optional 64-hex content hash (validated by :func:`validate_batch`).
    """

    token: str
    split_tag: str | None = None
    sha256: str | None = None

    def as_mapping(self) -> dict[str, Any]:
        """JSON-friendly view (``None`` fields omitted)."""
        return {key: value for key, value in {
            "token": self.token, "split_tag": self.split_tag,
            "sha256": self.sha256}.items() if value is not None}


@dataclass
class BaseBatch:
    """Base (identity/provenance) layer of the data-batch contract."""

    sample_id: Any | None = None
    targets: Any | None = None
    split_tag: Any | None = None
    provenance: Any | None = None

    def as_mapping(self, *, include_none: bool = False) -> dict[str, Any]:
        """Return the batch as a plain ``dict`` (non-``None`` fields by default)."""
        out: dict[str, Any] = {}
        for spec in dataclass_fields(self):
            value = getattr(self, spec.name)
            if include_none or value is not None:
                out[spec.name] = value
        return out

    def field_names(self) -> tuple[str, ...]:
        """All canonical field names of this batch class (order preserved)."""
        return tuple(spec.name for spec in dataclass_fields(self))


@dataclass
class SequenceBatch(BaseBatch):
    """Legacy V5.6 trajectory batch (compat layer, mapped 1:1)."""

    frame_feats: Any | None = None
    static_feats: Any | None = None
    lengths: Any | None = None
    delta_s: Any | None = None
    move_target: Any | None = None
    process_target: Any | None = None
    progress_target: Any | None = None
    trust_target: Any | None = None


@dataclass
class GraphBatch(BaseBatch):
    """Geometric data batch (graph layer)."""

    pos: Any | None = None
    z: Any | None = None
    edge_index: Any | None = None
    endpoint_mask: Any | None = None
    sym_pair_idx: Any | None = None
    target_xyz: Any | None = None
    sample_weight: Any | None = None


@dataclass
class StandardBatch(SequenceBatch, GraphBatch):
    """Deprecated v1 flat batch (graph + sequence fields) — compatibility only."""

    def __post_init__(self) -> None:
        warnings.warn(
            "StandardBatch is deprecated since contract v2; use GraphBatch or "
            "SequenceBatch (and validate_batch), or the to_standard/from_standard "
            "compat shim",
            DeprecationWarning, stacklevel=2)
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:  # pragma: no cover - no parent post-init today
            parent()


def _mapping_of(batch: Any) -> dict[str, Any]:
    if isinstance(batch, BaseBatch):
        return batch.as_mapping()
    if isinstance(batch, Mapping):
        return dict(batch)
    raise ContractViolation(
        "BAD_BATCH_TYPE",
        f"expected a batch object or mapping, got {type(batch).__name__!r}")


def to_standard(batch_dict: Any) -> SequenceBatch:
    """Deprecated: map a legacy V5.6 trajectory batch into a ``SequenceBatch``.

    Legacy keys are renamed 1:1 (``process_label`` -> ``process_target``,
    ``process_progress`` -> ``progress_target``, ``trust_label`` ->
    ``trust_target``, single-sample ``length`` -> ``lengths``,
    ``provenance_id`` -> ``provenance``).  Base identity fields pass through
    when present.  Extra legacy keys that are not part of the contract (e.g.
    ``frame_mask``, ``move_mask``, ``q``, ``s``) are NOT carried by the returned
    batch: masks are owned by the training collate layer, not by the contract.

    Use :class:`SequenceBatch` directly in new code.
    """
    warnings.warn(
        "to_standard() is deprecated since contract v2; construct SequenceBatch "
        "directly", DeprecationWarning, stacklevel=2)
    mapping = _mapping_of(batch_dict)
    for key in mapping:
        if is_truth_field(key):
            raise ContractViolation(
                "TRUTH_FIELD_FORBIDDEN",
                f"ground-truth field {key!r} is not allowed in a batch (R1)",
                str(key))
    renamed: dict[str, Any] = {}
    for key, value in mapping.items():
        renamed[LEGACY_TO_STANDARD.get(key, key)] = value
    known = set(BaseBatch.__dataclass_fields__) | set(SequenceBatch.__dataclass_fields__)
    return SequenceBatch(**{key: value for key, value in renamed.items()
                            if key in known})


def from_standard(batch: Any) -> dict[str, Any]:
    """Deprecated: map a batch back to the legacy V5.6 trajectory key names."""
    warnings.warn(
        "from_standard() is deprecated since contract v2; the v5 engine owns "
        "legacy batch collation", DeprecationWarning, stacklevel=2)
    mapping = _mapping_of(batch)
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        out[STANDARD_TO_LEGACY.get(key, key)] = value
    return out
