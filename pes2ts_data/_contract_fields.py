"""Field-level primitives of the 3-layer data-batch contract (v2).

Internal module (public surface: :mod:`pes2ts_data.contracts`).  Holds the typed
violation, truth-key detection, numpy coercion and per-field dtype/shape/
provenance checks; :mod:`pes2ts_data._contract_validation` composes them into
:func:`validate_batch`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Final

import numpy as np

from ._contract_specs import TRUTH_FIELD_PATTERNS

__all__ = [
    "ContractViolation",
    "check_value",
    "coerce_array",
    "forbid_truth_fields",
    "is_truth_field",
    "node_counts",
    "provenance_token",
]

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-fA-F]{64}$")


class ContractViolation(Exception):
    """Typed, honest rejection of a batch (P2).

    Attributes
    ----------
    code:
        Stable machine code from :data:`pes2ts_data._contract_specs.CODES`.
    detail:
        Human-readable explanation; never empty.
    field:
        Optional offending field name (``None`` for whole-batch problems).
    """

    def __init__(self, code: str, detail: str, field: str | None = None) -> None:
        self.code = str(code)
        self.detail = str(detail)
        self.field = field
        label = f" [{field}]" if field is not None else ""
        super().__init__(f"{self.code}{label}: {self.detail}")


def _normalise_key(name: object) -> str:
    return str(name).strip().lower().replace("-", "_").replace(".", "_").replace(" ", "_")


def is_truth_field(name: object) -> bool:
    """Return ``True`` when ``name`` looks like ground-truth material (R1)."""
    if not isinstance(name, str):
        return False
    key = _normalise_key(name)
    return any(pattern in key for pattern in TRUTH_FIELD_PATTERNS)


def _torch_to_numpy(value: Any) -> Any:
    """Best-effort torch -> numpy conversion without importing torch."""
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        try:
            return value.detach().cpu().numpy()
        except Exception:  # pragma: no cover - exotic tensor subclasses
            return value
    return value


def _as_array(value: Any, field: str) -> np.ndarray:
    value = _torch_to_numpy(value)
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value)
        except Exception as exc:  # pragma: no cover - ragged input
            raise ContractViolation("BAD_DTYPE", f"cannot interpret as array: {exc}",
                                    field) from exc
    if np.isscalar(value):
        return np.asarray(value)
    raise ContractViolation("BAD_DTYPE",
                            f"unsupported value type {type(value).__name__!r}", field)


def coerce_array(value: Any, field: str) -> np.ndarray:
    """Coerce a batch value (numpy/list/torch) to a numpy array."""
    return _as_array(value, field)


def _dtype_ok(spec_dtype: str, array: np.ndarray) -> bool:
    kind = array.dtype.kind
    if spec_dtype == "float":
        return kind == "f"
    if spec_dtype == "int":
        return kind in "iu"
    if spec_dtype == "bool":
        return kind == "b"
    if spec_dtype == "str":
        return kind in "USO"
    return spec_dtype == "any"


def _shape_ok(spec: Any, shape: tuple[int, ...]) -> bool:
    if not spec.accepts_ndim(len(shape)):
        return False
    return all(not isinstance(expected, int) or shape[axis] == expected
               for axis, expected in enumerate(spec.shape))


def _string_shape(spec: Any, value: Any) -> tuple[int, ...]:
    value = _torch_to_numpy(value)
    if isinstance(value, str):
        return ()
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "USO":
            raise ContractViolation("BAD_DTYPE",
                                    f"expected str array, got {value.dtype}", spec.name)
        return tuple(int(dim) for dim in value.shape)
    if isinstance(value, (list, tuple)):
        if not all(isinstance(item, str) for item in value):
            raise ContractViolation("BAD_DTYPE", "expected str or list[str]", spec.name)
        return (len(value),)
    raise ContractViolation("BAD_DTYPE",
                            f"expected str or list[str], got {type(value).__name__!r}",
                            spec.name)


def check_value(spec: Any, value: Any) -> dict[str, Any]:
    """Per-field dtype/shape check, returning its report entry."""
    if spec.dtype == "mapping":
        if not isinstance(value, Mapping):
            raise ContractViolation("TARGETS_INVALID",
                                    f"targets must be a mapping, got "
                                    f"{type(value).__name__!r}", spec.name)
        return {"dtype": "mapping", "shape": (len(value),),
                "provenance_sensitive": spec.provenance_sensitive}
    if spec.dtype == "any":
        return {"dtype": type(value).__name__, "shape": (),
                "provenance_sensitive": spec.provenance_sensitive}
    if spec.dtype == "str":
        shape = _string_shape(spec, value)
        if not spec.accepts_ndim(len(shape)):
            raise ContractViolation("BAD_SHAPE",
                                    f"expected ndim in {spec.ndim}, got shape {shape}",
                                    spec.name)
        return {"dtype": "str", "shape": shape,
                "provenance_sensitive": spec.provenance_sensitive}
    array = _as_array(value, spec.name)
    if not _dtype_ok(spec.dtype, array):
        raise ContractViolation("BAD_DTYPE",
                                f"expected dtype family {spec.dtype!r}, got "
                                f"{array.dtype}", spec.name)
    if not _shape_ok(spec, tuple(int(dim) for dim in array.shape)):
        raise ContractViolation(
            "BAD_SHAPE",
            f"expected shape template {spec.shape or spec.ndim}, got {array.shape}",
            spec.name)
    return {"dtype": str(array.dtype),
            "shape": tuple(int(dim) for dim in array.shape),
            "provenance_sensitive": spec.provenance_sensitive}


def provenance_token(value: Any) -> str:
    """Normalize a provenance value to its stable token (R1)."""
    if isinstance(value, str):
        if not value.strip():
            raise ContractViolation("PROVENANCE_INVALID", "provenance token is empty",
                                    "provenance")
        return value
    token = getattr(value, "token", None)
    sha = getattr(value, "sha256", None)
    if token is None and isinstance(value, Mapping):
        token, sha = value.get("token"), value.get("sha256")
    if not isinstance(token, str) or not token.strip():
        raise ContractViolation(
            "PROVENANCE_INVALID",
            "provenance must be a non-empty str, a Provenance or a mapping with "
            "a non-empty 'token'", "provenance")
    if sha is not None and not _SHA256_RE.match(str(sha)):
        raise ContractViolation("PROVENANCE_INVALID",
                                f"provenance sha256 is not 64 hex chars: {sha!r}",
                                "provenance")
    return token


def node_counts(arrays: dict[str, np.ndarray]) -> int | None:
    """Cross-field node-axis consistency check (graph layer)."""
    counts = {name: int(arrays[name].shape[0])
              for name in ("pos", "z", "endpoint_mask", "target_xyz", "sample_weight")
              if name in arrays}
    if len(set(counts.values())) > 1:
        detail = ", ".join(f"{name}={size}" for name, size in counts.items())
        raise ContractViolation("SHAPE_MISMATCH",
                                f"node-axis length differs across fields: {detail}",
                                next(iter(counts)))
    return next(iter(counts.values())) if counts else None


def forbid_truth_fields(batch: Any) -> None:
    """Refuse any truth-like key/field name in ``batch`` (R1)."""
    from .contracts import BaseBatch

    if isinstance(batch, BaseBatch):
        keys: Any = (field.name for field in batch.__dataclass_fields__.values())
    elif isinstance(batch, Mapping):
        keys = batch.keys()
    else:
        raise ContractViolation(
            "BAD_BATCH_TYPE",
            f"expected a batch object or mapping, got {type(batch).__name__!r}")
    for key in keys:
        if is_truth_field(key):
            raise ContractViolation(
                "TRUTH_FIELD_FORBIDDEN",
                f"ground-truth field {key!r} is not allowed in a training/"
                "inference batch (R1); read truth only through the audited "
                "truth bridge", str(key))
