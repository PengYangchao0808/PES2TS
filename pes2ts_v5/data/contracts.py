"""Lazy contract boundary to the data-zone batch contract (``pes2ts_data.contracts``).

The engine must not keep a private copy of the batch contract.  This module is a
thin, lazy adapter over the data package:

* preferred backend: contract v2 ``SequenceBatch`` (per-sample trajectory batches
  with ``sample_id`` + ``provenance`` identity, R1);
* while contract v2 is not on disk, the installed v1 ``StandardBatch`` is used as
  a documented compatibility backend (:func:`contract_backend` reports which one);
* without any installed contract, functions raise the typed
  :class:`DataContractUnavailable` (``code="DATA_CONTRACT_UNAVAILABLE"``) instead
  of silently skipping validation.

Validation always goes through the data package:
``assert_fields(batch, model.REQUIRED_FIELDS)`` for field presence and
``validate_batch(batch, require_core=False)`` for dtype/shape/identity/provenance
(P2/R1).  A *batched* legacy mapping is emitted as one contract batch **per
sample**, so each trajectory carries its own ``provenance`` token.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch

#: prefix that selects the real-data ``idx_v2`` store in ``data.store_manifest``
IDX_V2_PREFIX = "idx_v2:"
#: contract fields carrying batch-level identity (not per-sample tensors)
IDENTITY_FIELDS = ("sample_id", "targets", "split_tag", "provenance")


class DataError(RuntimeError):
    """Base class for typed V5 data-layer failures (``code`` is stable)."""

    code = "DATA_ERROR"

    def __init__(self, message: str, *, code: str | None = None):
        self.message = str(message)
        if code is not None:
            self.code = str(code)
        super().__init__(f"[{self.code}] {self.message}")


class DataContractUnavailable(DataError):
    """``pes2ts_data`` (or its contract / readers) is not installed."""

    code = "DATA_CONTRACT_UNAVAILABLE"


class DataSourceUnreachable(DataError):
    """A configured real-data source exists in the config but not on this host."""

    code = "DATA_SOURCE_UNREACHABLE"


class DataContractError(DataError):
    """A batch/loader value violated the shared contract shape."""

    code = "DATA_CONTRACT_INVALID"


# ---------------------------------------------------------------------------
# Lazy access to the shared contract
# ---------------------------------------------------------------------------

def _contract_module():
    try:
        import pes2ts_data.contracts as contracts
    except ImportError as exc:  # keep the engine importable without the package
        raise DataContractUnavailable(
            "pes2ts_data.contracts is not importable; the V5.6 engine requires "
            "the shared data contract (install pes2ts_data)") from exc
    return contracts


def contract_version() -> str:
    """Version of the installed batch contract (``""`` when unavailable)."""
    try:
        return str(_contract_module().CONTRACT_VERSION)
    except DataContractUnavailable:
        return ""


def _batch_class_name(module: Any) -> str:
    for name in ("SequenceBatch", "StandardBatch"):
        if hasattr(module, name):
            return name
    return "unavailable"


def contract_backend() -> str:
    """``"SequenceBatch"`` (contract v2), ``"StandardBatch"`` (v1 compat) or ``"unavailable"``."""
    try:
        return _batch_class_name(_contract_module())
    except DataContractUnavailable:
        return "unavailable"


def contracts_available() -> bool:
    """True when a usable batch contract (v2 or the v1 compat backend) is installed."""
    return contract_backend() != "unavailable"


def _symbols() -> dict[str, Any]:
    module = _contract_module()
    name = _batch_class_name(module)
    if name == "unavailable":
        raise DataContractUnavailable(
            "pes2ts_data.contracts exposes neither SequenceBatch (contract v2) nor "
            "StandardBatch (v1 compat); cannot validate batches")
    required = ("assert_fields", "validate_batch")
    missing = [attr for attr in required if not hasattr(module, attr)]
    if missing:
        raise DataContractUnavailable(
            f"pes2ts_data.contracts is missing {missing}; cannot validate batches")
    return {
        "backend": name,
        "version": str(getattr(module, "CONTRACT_VERSION", "")),
        "batch_cls": getattr(module, name),
        "assert_fields": getattr(module, "assert_fields"),
        "validate_batch": getattr(module, "validate_batch"),
        "legacy_to_standard": dict(getattr(module, "LEGACY_TO_STANDARD", {}) or {}),
        "standard_to_legacy": dict(getattr(module, "STANDARD_TO_LEGACY", {}) or {}),
        "Provenance": getattr(module, "Provenance", None),
        "is_truth_field": getattr(module, "is_truth_field", None),
        "ContractViolation": getattr(module, "ContractViolation", Exception),
    }


def contract_violation_type() -> type[Exception]:
    """The contract's typed ``ContractViolation`` class (lazy)."""
    return _symbols()["ContractViolation"]


# ---------------------------------------------------------------------------
# Batch construction / validation
# ---------------------------------------------------------------------------

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


def _field_list(batch_cls: type) -> list[str]:
    return ([spec.name for spec in dataclass_fields(batch_cls)]
            if is_dataclass(batch_cls) else [])


def _standard_mapping(source: Mapping[str, Any], legacy_map: Mapping[str, str]
                      ) -> dict[str, Any]:
    """Rename legacy V5 keys to contract vocabulary (no deprecated shims)."""
    return {legacy_map.get(key, key): value for key, value in source.items()}


def _slice_sample(value: Any, index: int, batch_size: int, name: str) -> Any:
    """Take one sample out of a batch tensor (``lengths`` keeps its 1-D shape)."""
    if not torch.is_tensor(value) or value.dim() < 1 or int(value.shape[0]) != batch_size:
        return value
    if name == "lengths":
        return value[index:index + 1]
    return value[index]


def to_sequence_batches(
    batch: Mapping[str, Any] | Any,
    *,
    provenance_id: Iterable[str] | Sequence[str] | None = None,
) -> list[Any]:
    """Build contract batch object(s) from a legacy V5 trajectory mapping.

    * contract v2 (``SequenceBatch``): one batch object **per sample**, each with
      its own ``sample_id`` + ``provenance`` token (scalar identity as required by
      contract v2), and per-sample compat fields;
    * v1 compat (``StandardBatch``): one object carrying the whole mapping and the
      per-sample ``provenance_id`` list.

    Extra legacy keys (``frame_mask``, ``move_mask`` ...) stay in the model/loss
    transport mapping and are not part of the contract batch.
    """
    symbols = _symbols()
    if isinstance(batch, Mapping):
        source = dict(batch)
    elif hasattr(batch, "as_mapping"):
        source = dict(batch.as_mapping())
    else:
        raise DataContractError(
            f"expected a batch mapping, got {type(batch).__name__!r}")
    field_names = _field_list(symbols["batch_cls"])
    if not field_names:
        raise DataContractUnavailable(
            f"contract batch class {symbols['batch_cls']!r} is not a dataclass; "
            "cannot construct a sequence batch")
    standard = _standard_mapping(source, symbols["legacy_to_standard"])
    # R1 pre-check on the *transport* mapping: keys outside the contract batch
    # class are dropped during construction, so ground-truth keys must be rejected
    # before that (reuse the contract's own detector, never a private copy).
    is_truth_field = symbols["is_truth_field"]
    if is_truth_field is not None:
        for key in source:
            if is_truth_field(key):
                raise symbols["ContractViolation"](
                    "TRUTH_FIELD_FORBIDDEN",
                    f"ground-truth field {key!r} is not allowed in a training/"
                    "inference batch (R1)", str(key))
    tokens = list(provenance_id) if provenance_id is not None else None

    legacy_backend = ("provenance_id" in field_names
                      and "provenance" not in field_names)
    if legacy_backend:
        kwargs = {name: standard[name] for name in field_names if name in standard}
        if tokens is not None:
            kwargs["provenance_id"] = tokens
        try:
            return [symbols["batch_cls"](**kwargs)]
        except TypeError as exc:  # pragma: no cover - v1 backend only
            raise DataContractError(
                f"cannot build {symbols['backend']}: {exc}") from exc

    samples = _batch_size(standard) or 1
    if tokens is None:
        tokens = [f"traj:index-{index}" for index in range(samples)]
    if len(tokens) != samples:
        raise DataContractError(
            f"provenance token count {len(tokens)} != batch size {samples}")
    provenance_cls = symbols["Provenance"]
    built: list[Any] = []
    for index, token in enumerate(tokens):
        kwargs: dict[str, Any] = {}
        for name in field_names:
            if name == "sample_id":
                kwargs[name] = token
            elif name == "provenance":
                kwargs[name] = (provenance_cls(token=token)
                                if provenance_cls is not None else token)
            elif name in IDENTITY_FIELDS:
                continue
            elif name in standard:
                kwargs[name] = _slice_sample(standard[name], index, samples, name)
        try:
            built.append(symbols["batch_cls"](**kwargs))
        except TypeError as exc:
            raise DataContractError(f"cannot build {symbols['backend']}: {exc}") from exc
    return built


def make_sequence_batch(
    batch: Mapping[str, Any] | Any,
    *,
    provenance_id: Iterable[str] | str | None = None,
) -> Any:
    """Single contract batch (first sample for batched v2 mappings).

    Prefer :func:`to_sequence_batches` for whole batches — contract v2 identity is
    per sample, so one mapping of ``B`` trajectories maps to ``B`` objects.
    """
    tokens: Any = provenance_id
    if isinstance(tokens, str):
        tokens = [tokens]
    built = to_sequence_batches(batch, provenance_id=tokens)
    return built[0]


def validate_sequence_batch(batch: Any, required_fields: Iterable[str] = (), *,
                            require_core: bool = False) -> dict[str, Any]:
    """Validate a contract batch: ``assert_fields`` then ``validate_batch``.

    ``required_fields`` is normally ``model.REQUIRED_FIELDS``; ``require_core``
    stays ``False`` because pure trajectory batches carry no ``pos``/``z``/
    ``edge_index`` geometric core.
    """
    symbols = _symbols()
    required = tuple(str(name) for name in required_fields)
    symbols["assert_fields"](batch, required)
    return symbols["validate_batch"](batch, required=required,
                                     require_core=require_core)


def legacy_mapping(batch: Any) -> dict[str, Any]:
    """Return the legacy V5 key mapping of a contract batch."""
    if isinstance(batch, Mapping):
        mapping = dict(batch)
    elif hasattr(batch, "as_mapping"):
        mapping = dict(batch.as_mapping())
    elif is_dataclass(batch):
        mapping = {spec.name: getattr(batch, spec.name)
                   for spec in dataclass_fields(batch)
                   if getattr(batch, spec.name) is not None}
    else:
        raise DataContractError(
            f"expected a contract batch or mapping, got {type(batch).__name__!r}")
    standard_to_legacy = _symbols()["standard_to_legacy"]
    return {standard_to_legacy.get(key, key): value
            for key, value in mapping.items()}


def assert_batch_fields(batch: Any, required: Iterable[str]) -> None:
    """Raise the typed ``MISSING_FIELD``/``TRUTH_FIELD_FORBIDDEN`` violation."""
    _symbols()["assert_fields"](batch, tuple(str(name) for name in required))


__all__ = [
    "IDX_V2_PREFIX",
    "IDENTITY_FIELDS",
    "DataContractError",
    "DataContractUnavailable",
    "DataError",
    "DataSourceUnreachable",
    "assert_batch_fields",
    "contract_backend",
    "contract_version",
    "contract_violation_type",
    "contracts_available",
    "legacy_mapping",
    "make_sequence_batch",
    "provenance_tokens",
    "to_sequence_batches",
    "validate_sequence_batch",
]
