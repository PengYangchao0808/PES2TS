"""The single data-batch contract for PES2TS (Zone 2).

This module is the *only* place that declares the canonical batch field set,
per-field dtype/shape/provenance metadata and the typed rejection codes used by
every loader, trainer and evaluator.  Consumers pass a :class:`StandardBatch`
(or a plain mapping with the same keys) to :func:`validate_batch` instead of
re-inventing ad-hoc dictionaries.

Constitution mapping
--------------------
R1  truth isolation
    Any batch key whose name suggests ground truth (``truth``, ``ts_geometry``,
    ``ground_truth`` ...) is rejected with ``TRUTH_FIELD_FORBIDDEN``.  Fields
    that are derived from the audited teacher/truth path (``target_xyz``,
    ``sample_weight`` and the compat targets) are declared
    ``provenance_sensitive`` and require ``provenance_id``; otherwise
    ``PROVENANCE_MISSING`` is raised.
R5  full-cost retention
    ``sample_weight`` exists so every attempted position/trajectory carries its
    cost weight into the batch; batches must not silently drop failed attempts.
R6  split firewall
    ``provenance_id`` carries the ``(curve_id, split, ...)`` lineage; the split
    side of R6 lives in :mod:`pes2ts_data.splits.policy`.  This contract never
    merges splits.
P2  typed honest rejection
    Every violation raises :class:`ContractViolation` with a *stable* machine
    code -- never ``KeyError``, ``TypeError`` or silent imputation.

Dependencies
------------
Only the standard library (:mod:`dataclasses`, :mod:`typing`) and
:mod:`numpy` are imported at module level.  Torch tensors are accepted through
duck typing (``.detach().cpu().numpy()``) without importing torch; pandas /
pyarrow are never needed here.

Field groups
------------
* ``canonical`` -- the geometric batch identity: ``pos``, ``z``,
  ``edge_index``, ``endpoint_mask``, ``sym_pair_idx``, ``target_xyz``,
  ``sample_weight``, ``provenance_id``.
* ``compat`` -- the legacy V5.6 trajectory-dataset keys kept as a 1:1 bridge so
  the existing dataset/collate code keeps working while the rest of the repo
  migrates; see :data:`LEGACY_FIELD_MAP` and :func:`to_standard` /
  :func:`from_standard`.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = [
    "CODES",
    "CONTRACT_VERSION",
    "COMPAT_FIELDS",
    "CONTRACT_REQUIRED_FIELDS",
    "FIELD_SPECS",
    "LEGACY_FIELD_MAP",
    "LEGACY_TO_STANDARD",
    "STANDARD_TO_LEGACY",
    "TRUTH_FIELD_PATTERNS",
    "ContractViolation",
    "FieldSpec",
    "StandardBatch",
    "assert_fields",
    "from_standard",
    "is_truth_field",
    "required_fields",
    "to_standard",
    "validate_batch",
]

#: Version of the batch contract (bump on any breaking field change, P1).
CONTRACT_VERSION = "1.0.0"

#: Stable machine-readable violation codes.  New codes are appended, never
#: renamed, so downstream handlers can branch on ``.code`` forever.
CODES = (
    "MISSING_FIELD",
    "BAD_DTYPE",
    "BAD_SHAPE",
    "BAD_VALUE",
    "BAD_INDEX",
    "BAD_BATCH_TYPE",
    "UNKNOWN_FIELD",
    "SHAPE_MISMATCH",
    "PROVENANCE_MISSING",
    "TRUTH_FIELD_FORBIDDEN",
)

#: Substrings that mark a key as ground-truth material (R1).  Matching is
#: case-insensitive on the normalised key (``-``/``.``/spaces -> ``_``).
TRUTH_FIELD_PATTERNS = (
    "truth",
    "ground_truth",
    "groundtruth",
    "ts_geometry",
    "ts_geom",
    "ts_xyz",
    "ts_coords",
    "reference_ts",
    "true_ts",
    "ref_ts_",
)

#: dtype families accepted by :class:`FieldSpec`.
DTYPE_FAMILIES = ("float", "int", "bool", "str", "any")


class ContractViolation(Exception):
    """Typed, honest rejection of a batch (P2).

    Attributes
    ----------
    code:
        Stable machine code from :data:`CODES` (e.g. ``MISSING_FIELD``).
    detail:
        Human-readable explanation; never empty.
    field:
        Optional offending field name (``None`` for whole-batch problems).
    """

    def __init__(self, code: str, detail: str, field: str | None = None) -> None:
        self.code = str(code)
        self.detail = str(detail)
        self.field = field
        message = f"{self.code}: {self.detail}"
        if field is not None and f"[{field}]" not in self.detail:
            message = f"{self.code} [{field}]: {self.detail}"
        super().__init__(message)


@dataclass(frozen=True)
class FieldSpec:
    """Machine-readable metadata for one batch field.

    Parameters
    ----------
    name:
        Canonical field name (the key used everywhere downstream).
    dtype:
        dtype family: one of :data:`DTYPE_FAMILIES`.
    ndim:
        Accepted number of dimensions: an ``int`` or a tuple of ints.
    required:
        ``True`` when the field belongs to the minimal contract batch that can
        be fed to a geometric model (``pos``/``z``/``edge_index``); enforced by
        :func:`validate_batch` unless ``require_core=False``.
    description:
        One-line human description of the field semantics.
    provenance_sensitive:
        ``True`` when the values are derived from the audited teacher/truth
        path (R1): ``provenance_id`` is then mandatory.
    mutable:
        ``False`` for labels/targets/provenance that downstream code must treat
        as read-only (in-place edits invalidate provenance).
    shape:
        Optional shape template.  ``int`` entries are enforced exactly;
        symbolic string entries (``"N"`` nodes, ``"E"`` edges, ``"B"`` batch,
        ``"T"`` frames, ``"P"`` pairs) are used only for cross-field consistency.
    compat:
        ``True`` for the legacy V5.6 trajectory compat fields.
    """

    name: str
    dtype: str
    ndim: int | tuple[int, ...]
    required: bool
    description: str
    provenance_sensitive: bool = False
    mutable: bool = True
    shape: tuple[int | str | None, ...] = ()
    compat: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in DTYPE_FAMILIES:
            raise ValueError(f"FieldSpec({self.name!r}): unknown dtype {self.dtype!r}")
        ndims = self.ndim if isinstance(self.ndim, tuple) else (self.ndim,)
        if not ndims or any(not isinstance(n, int) or n < 0 for n in ndims):
            raise ValueError(f"FieldSpec({self.name!r}): bad ndim {self.ndim!r}")
        if self.shape and len(self.shape) not in (0, *ndims):
            raise ValueError(
                f"FieldSpec({self.name!r}): shape template length {len(self.shape)} "
                f"not in accepted ndims {ndims}")

    def accepts_ndim(self, ndim: int) -> bool:
        """Return ``True`` when ``ndim`` is an accepted rank."""
        ndims = self.ndim if isinstance(self.ndim, tuple) else (self.ndim,)
        return ndim in ndims


# ---------------------------------------------------------------------------
# Canonical batch
# ---------------------------------------------------------------------------

@dataclass
class StandardBatch:
    """The single canonical data batch.

    Every field is optional at the dataclass level; the *model* declares its own
    ``REQUIRED_FIELDS`` and the loader enforces them with :func:`assert_fields`
    / :func:`validate_batch` (P2).  The minimal geometric contract fields
    ``pos``, ``z`` and ``edge_index`` are marked ``required`` in
    :data:`FIELD_SPECS`.

    Canonical fields
    ---------------
    pos:
        ``(N, 3)`` float atomic coordinates (Angstrom).
    z:
        ``(N,)`` integer atomic numbers.
    edge_index:
        ``(2, E)`` integer directed edges, ``[row=receiver, col=sender]``.
    endpoint_mask:
        ``(N,)`` bool node mask or ``(B,)`` bool batch-level mask telling the
        loss which endpoint/atom observations are usable.
    sym_pair_idx:
        ``(P, 2)`` integer symmetric pair indices (canonical key-atom pairs).
    target_xyz:
        ``(N, 3)`` float supervised displacement/target geometry, teacher
        derived -> provenance required (R1).
    sample_weight:
        ``(N,)`` float per-sample cost weight (R5; never silently dropped).
    provenance_id:
        ``str`` or list of ``str`` provenance token(s); required whenever a
        ``provenance_sensitive`` field is present.

    Compat fields (legacy V5.6 trajectory dataset, mapped 1:1)
    ----------------------------------------------------------
    ``frame_feats``, ``static_feats``, ``lengths``, ``delta_s``,
    ``move_target``, ``process_target``, ``progress_target``, ``trust_target``
    are the legacy trajectory keys renamed to the contract vocabulary.  See
    :func:`to_standard` / :func:`from_standard` and :data:`LEGACY_FIELD_MAP`.
    Extra legacy keys (``frame_mask``, ``move_mask``, ``q``, ``s`` ...) are
    passed through untouched by the compat adapters and ignored by validation.
    """

    # -- canonical geometric batch ------------------------------------------
    pos: Any | None = None
    z: Any | None = None
    edge_index: Any | None = None
    endpoint_mask: Any | None = None
    sym_pair_idx: Any | None = None
    target_xyz: Any | None = None
    sample_weight: Any | None = None
    provenance_id: Any | None = None
    # -- legacy V5.6 trajectory compat (1:1 bridge) -------------------------
    frame_feats: Any | None = None
    static_feats: Any | None = None
    lengths: Any | None = None
    delta_s: Any | None = None
    move_target: Any | None = None
    process_target: Any | None = None
    progress_target: Any | None = None
    trust_target: Any | None = None

    def as_mapping(self, *, include_none: bool = False) -> dict[str, Any]:
        """Return the batch as a plain ``dict`` (non-``None`` fields by default)."""
        out: dict[str, Any] = {}
        for spec in dataclass_fields(self):
            value = getattr(self, spec.name)
            if include_none or value is not None:
                out[spec.name] = value
        return out

    def field_names(self) -> tuple[str, ...]:
        """Return all canonical dataclass field names (order preserved)."""
        return tuple(spec.name for spec in dataclass_fields(self))


# ---------------------------------------------------------------------------
# Field specs
# ---------------------------------------------------------------------------

FIELD_SPECS: dict[str, FieldSpec] = {
    # -- canonical ----------------------------------------------------------
    "pos": FieldSpec(
        name="pos", dtype="float", ndim=2, required=True,
        description="(N,3) float atomic coordinates in Angstrom.",
        shape=("N", 3)),
    "z": FieldSpec(
        name="z", dtype="int", ndim=1, required=True,
        description="(N,) integer atomic numbers.",
        shape=("N",)),
    "edge_index": FieldSpec(
        name="edge_index", dtype="int", ndim=2, required=True,
        description="(2,E) integer edges [row=receiver, col=sender].",
        shape=(2, "E")),
    "endpoint_mask": FieldSpec(
        name="endpoint_mask", dtype="bool", ndim=(1, 2), required=False,
        description="(N,) node-level or (B,) batch-level usable-endpoint mask.",
        shape=("N",)),
    "sym_pair_idx": FieldSpec(
        name="sym_pair_idx", dtype="int", ndim=2, required=False,
        description="(P,2) integer canonical symmetric key-atom pairs.",
        shape=("P", 2)),
    "target_xyz": FieldSpec(
        name="target_xyz", dtype="float", ndim=2, required=False,
        description="(N,3) float teacher-derived target geometry; provenance required (R1).",
        provenance_sensitive=True, mutable=False, shape=("N", 3)),
    "sample_weight": FieldSpec(
        name="sample_weight", dtype="float", ndim=1, required=False,
        description="(N,) float per-sample cost weight; full-cost retention (R5).",
        provenance_sensitive=True, mutable=False, shape=("N",)),
    "provenance_id": FieldSpec(
        name="provenance_id", dtype="str", ndim=(0, 1), required=False,
        description="Provenance token: str or list[str] (curve/split/sha lineage).",
        mutable=False),
    # -- legacy V5.6 trajectory compat (mapped 1:1) -------------------------
    "frame_feats": FieldSpec(
        name="frame_feats", dtype="float", ndim=(2, 3), required=False,
        description="Compatibility: legacy frame features [T,F] / [B,T,F].",
        provenance_sensitive=True, compat=True),
    "static_feats": FieldSpec(
        name="static_feats", dtype="float", ndim=(1, 2), required=False,
        description="Compatibility: legacy static reaction features [S] / [B,S].",
        provenance_sensitive=True, compat=True),
    "lengths": FieldSpec(
        name="lengths", dtype="int", ndim=1, required=False,
        description="Compatibility: legacy per-trajectory lengths [B] (sample key 'length').",
        compat=True),
    "delta_s": FieldSpec(
        name="delta_s", dtype="float", ndim=(1, 2), required=False,
        description="Compatibility: legacy per-frame arc-length step delta_s.",
        compat=True),
    "move_target": FieldSpec(
        name="move_target", dtype="float", ndim=(2, 3), required=False,
        description="Compatibility: legacy MOVE target displacement [T,K] / [B,T,K].",
        provenance_sensitive=True, mutable=False, compat=True),
    "process_target": FieldSpec(
        name="process_target", dtype="int", ndim=(1, 2), required=False,
        description="Compatibility: legacy PROCESS class labels (legacy 'process_label').",
        provenance_sensitive=True, mutable=False, compat=True),
    "progress_target": FieldSpec(
        name="progress_target", dtype="float", ndim=(1, 2), required=False,
        description="Compatibility: legacy PROCESS progress (legacy 'process_progress').",
        provenance_sensitive=True, mutable=False, compat=True),
    "trust_target": FieldSpec(
        name="trust_target", dtype="float", ndim=(1, 2), required=False,
        description="Compatibility: legacy trust target (legacy 'trust_label').",
        provenance_sensitive=True, mutable=False, compat=True),
}

#: Fields that form the minimal contract batch (enforced by default).
CONTRACT_REQUIRED_FIELDS: tuple[str, ...] = tuple(
    name for name, spec in FIELD_SPECS.items() if spec.required)

#: Fields belonging to the legacy V5.6 trajectory compat bridge.
COMPAT_FIELDS: tuple[str, ...] = tuple(
    name for name, spec in FIELD_SPECS.items() if spec.compat)

#: Canonical <-> legacy V5.6 trajectory key mapping (see :data:`LEGACY_FIELD_MAP`).
LEGACY_FIELD_MAP: dict[str, str] = {
    # standard compat field -> legacy trajectory key
    "frame_feats": "frame_feats",
    "static_feats": "static_feats",
    "lengths": "lengths",           # single-sample legacy key is "length"
    "delta_s": "delta_s",
    "move_target": "move_target",
    "process_target": "process_label",
    "progress_target": "process_progress",
    "trust_target": "trust_label",
}

#: Legacy key -> standard field name (inverse of :data:`LEGACY_FIELD_MAP` plus
#: the single-sample ``length`` alias).
LEGACY_TO_STANDARD: dict[str, str] = {
    **{legacy: std for std, legacy in LEGACY_FIELD_MAP.items()},
    "length": "lengths",
}
STANDARD_TO_LEGACY: dict[str, str] = dict(LEGACY_FIELD_MAP)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Convert a batch value to ``numpy`` or raise ``BAD_DTYPE``."""
    value = _torch_to_numpy(value)
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value)
        except Exception as exc:  # pragma: no cover - ragged input
            raise ContractViolation(
                "BAD_DTYPE", f"cannot interpret as array: {exc}", field) from exc
    if np.isscalar(value):
        return np.asarray(value)
    raise ContractViolation(
        "BAD_DTYPE", f"unsupported value type {type(value).__name__!r}", field)


def _dtype_ok(spec: FieldSpec, array: np.ndarray) -> bool:
    kind = array.dtype.kind
    if spec.dtype == "float":
        return kind == "f"
    if spec.dtype == "int":
        return kind in "iu"
    if spec.dtype == "bool":
        return kind == "b"
    if spec.dtype == "any":
        return True
    if spec.dtype == "str":
        return kind in "USO"
    return False


def _shape_ok(spec: FieldSpec, shape: tuple[int, ...]) -> bool:
    if not spec.accepts_ndim(len(shape)):
        return False
    for axis, expected in enumerate(spec.shape):
        if isinstance(expected, int) and shape[axis] != expected:
            return False
    return True


def _validate_string_field(spec: FieldSpec, value: Any) -> tuple[int, ...]:
    """Validate a ``str``-family field and return its shape."""
    value = _torch_to_numpy(value)
    if isinstance(value, str):
        return ()
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "USO":
            raise ContractViolation(
                "BAD_DTYPE", f"expected str array, got dtype {value.dtype}", spec.name)
        return tuple(int(d) for d in value.shape)
    if isinstance(value, (list, tuple)):
        if not all(isinstance(item, str) for item in value):
            raise ContractViolation(
                "BAD_DTYPE", "expected str or list[str]", spec.name)
        return (len(value),)
    raise ContractViolation(
        "BAD_DTYPE", f"expected str or list[str], got {type(value).__name__!r}", spec.name)


def _batch_items(batch: Any) -> list[tuple[str, Any]]:
    """Return the available ``(key, value)`` pairs of a batch-like object."""
    if isinstance(batch, StandardBatch):
        return [(spec.name, getattr(batch, spec.name))
                for spec in dataclass_fields(batch) if getattr(batch, spec.name) is not None]
    if isinstance(batch, Mapping):
        return list(batch.items())
    raise ContractViolation(
        "BAD_BATCH_TYPE",
        f"expected StandardBatch or mapping, got {type(batch).__name__!r}")


def _available_keys(batch: Any) -> set[str]:
    if isinstance(batch, StandardBatch):
        return {spec.name for spec in dataclass_fields(batch)
                if getattr(batch, spec.name) is not None}
    if isinstance(batch, Mapping):
        return {str(key) for key in batch.keys() if isinstance(key, str)}
    raise ContractViolation(
        "BAD_BATCH_TYPE",
        f"expected StandardBatch or mapping, got {type(batch).__name__!r}")


def _forbid_truth_fields(batch: Any) -> None:
    if isinstance(batch, Mapping):
        keys: Iterable[object] = batch.keys()
    elif isinstance(batch, StandardBatch):
        keys = (spec.name for spec in dataclass_fields(batch))
    else:
        raise ContractViolation(
            "BAD_BATCH_TYPE",
            f"expected StandardBatch or mapping, got {type(batch).__name__!r}")
    for key in keys:
        if is_truth_field(key):
            raise ContractViolation(
                "TRUTH_FIELD_FORBIDDEN",
                f"ground-truth field {key!r} is not allowed in a training/inference "
                "batch (R1); read truth only through the audited truth reader",
                str(key))


# ---------------------------------------------------------------------------
# Public validation API
# ---------------------------------------------------------------------------

def required_fields(required: Iterable[str], available: Iterable[str]) -> list[str]:
    """Return the fields from ``required`` that are absent from ``available``.

    Order follows ``required``; duplicates are collapsed.  This is the P2 helper
    used by loaders to reject a model's unmet ``REQUIRED_FIELDS`` instead of
    raising ``KeyError`` deep inside the model.
    """
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
    """Raise ``MISSING_FIELD`` when a required field is absent from ``batch``.

    ``required`` is normally the model's ``REQUIRED_FIELDS`` tuple.  Truth-like
    names are rejected with ``TRUTH_FIELD_FORBIDDEN``: a model may not declare a
    ground-truth field as an input requirement (R1).
    """
    for name in required:
        if is_truth_field(name):
            raise ContractViolation(
                "TRUTH_FIELD_FORBIDDEN",
                f"required field {name!r} looks like ground truth; models may not "
                "require truth fields as input (R1)",
                str(name))
    available = _available_keys(batch)
    missing = required_fields(required, available)
    if missing:
        raise ContractViolation(
            "MISSING_FIELD",
            f"missing required batch field(s): {', '.join(missing)}; "
            f"available: {', '.join(sorted(available)) or '<none>'}",
            missing[0])


def validate_batch(
    batch: Any,
    required: Iterable[str] = (),
    *,
    require_core: bool = True,
    require_provenance: bool = True,
    allow_unknown: bool = True,
) -> dict[str, Any]:
    """Validate ``batch`` and return a normalised validation report.

    Parameters
    ----------
    batch:
        A :class:`StandardBatch` or a plain mapping with the same keys.
    required:
        Extra model-declared fields that must be present (``REQUIRED_FIELDS``).
    require_core:
        Enforce :data:`CONTRACT_REQUIRED_FIELDS` (``pos``/``z``/``edge_index``).
        Set ``False`` for pure legacy trajectory compat batches.
    require_provenance:
        Enforce that any ``provenance_sensitive`` field present is accompanied by
        ``provenance_id`` (R1).
    allow_unknown:
        When ``False``, keys outside :data:`FIELD_SPECS` raise ``UNKNOWN_FIELD``.
        Default is tolerant because the legacy V5.6 collate batch carries extra
        keys (``frame_mask``, ``move_mask`` ...) that the compat bridge passes
        through untouched.

    Returns
    -------
    dict
        ``{"fields": {name: {"dtype", "shape", "provenance_sensitive"}},
        "required": [...], "provenance_id": bool, "compat_fields": [...]}``.

    Raises
    ------
    ContractViolation
        With a stable ``.code``; never ``KeyError``/``TypeError``.
    """
    _forbid_truth_fields(batch)
    items = _batch_items(batch)
    present: dict[str, Any] = {}
    for key, value in items:
        if not isinstance(key, str):
            raise ContractViolation(
                "BAD_BATCH_TYPE", f"batch keys must be str, got {type(key).__name__!r}")
        present[key] = value

    unknown = sorted(name for name in present if name not in FIELD_SPECS)
    if unknown and not allow_unknown:
        raise ContractViolation(
            "UNKNOWN_FIELD",
            f"unknown batch field(s): {', '.join(unknown)}; "
            f"known: {', '.join(FIELD_SPECS)}",
            unknown[0])

    # -- declared required fields (P2) --------------------------------------
    effective_required = list(CONTRACT_REQUIRED_FIELDS if require_core else ())
    effective_required += [str(name) for name in required]
    assert_fields(batch, effective_required)

    # -- per-field dtype / shape -------------------------------------------
    report_fields: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    for name, value in present.items():
        spec = FIELD_SPECS.get(name)
        if spec is None:
            continue  # tolerated legacy passthrough key
        if spec.dtype == "str":
            shape = _validate_string_field(spec, value)
            if not spec.accepts_ndim(len(shape)):
                raise ContractViolation(
                    "BAD_SHAPE",
                    f"expected ndim in {spec.ndim}, got shape {shape}", name)
            report_fields[name] = {
                "dtype": "str", "shape": shape,
                "provenance_sensitive": spec.provenance_sensitive,
            }
            continue
        array = _as_array(value, name)
        if not _dtype_ok(spec, array):
            raise ContractViolation(
                "BAD_DTYPE",
                f"expected dtype family {spec.dtype!r}, got {array.dtype}",
                name)
        if not _shape_ok(spec, tuple(int(d) for d in array.shape)):
            raise ContractViolation(
                "BAD_SHAPE",
                f"expected shape template {spec.shape or spec.ndim}, got {array.shape}",
                name)
        arrays[name] = array
        report_fields[name] = {
            "dtype": str(array.dtype),
            "shape": tuple(int(d) for d in array.shape),
            "provenance_sensitive": spec.provenance_sensitive,
        }

    # -- cross-field consistency -------------------------------------------
    node_counts = {}
    for name in ("pos", "z", "endpoint_mask", "target_xyz", "sample_weight"):
        if name in arrays:
            node_counts[name] = int(arrays[name].shape[0])
    if len(set(node_counts.values())) > 1:
        detail = ", ".join(f"{name}={n}" for name, n in node_counts.items())
        raise ContractViolation(
            "SHAPE_MISMATCH",
            f"node-axis length differs across fields: {detail}",
            next(iter(node_counts)))
    n_atoms = next(iter(node_counts.values())) if node_counts else None

    if n_atoms is not None:
        for name in ("edge_index", "sym_pair_idx"):
            if name not in arrays:
                continue
            idx = arrays[name]
            if idx.size and (idx.min() < 0 or idx.max() >= n_atoms):
                raise ContractViolation(
                    "BAD_INDEX",
                    f"indices out of range [0,{n_atoms}) (min={int(idx.min())}, "
                    f"max={int(idx.max())})",
                    name)

    # -- R1 provenance requirement -----------------------------------------
    provenance_present = "provenance_id" in present
    if require_provenance:
        sensitive = [name for name in present
                     if name in FIELD_SPECS and FIELD_SPECS[name].provenance_sensitive]
        if sensitive and not provenance_present:
            raise ContractViolation(
                "PROVENANCE_MISSING",
                "provenance-sensitive field(s) present without provenance_id: "
                f"{', '.join(sorted(sensitive))} (R1); attach a provenance token "
                "or pass require_provenance=False only if the loader already "
                "audited provenance upstream",
                sorted(sensitive)[0])

    return {
        "fields": report_fields,
        "required": effective_required,
        "provenance_id": provenance_present,
        "compat_fields": sorted(name for name in present if name in COMPAT_FIELDS),
        "unknown_fields": unknown,
    }


# ---------------------------------------------------------------------------
# Legacy V5.6 trajectory compat bridge
# ---------------------------------------------------------------------------

def to_standard(batch_dict: Any) -> dict[str, Any]:
    """Translate a legacy V5.6 trajectory batch into standard field names.

    The legacy :mod:`pes2ts_v5.data.trajectories` dataset/collate code emits
    ``length``/``lengths``, ``process_label``, ``process_progress`` and
    ``trust_label``.  This adapter renames them to the compat field names
    declared in :data:`FIELD_SPECS` so the same batch can be validated:

    =========================  =========================
    legacy trajectory key      standard compat field
    =========================  =========================
    ``frame_feats``            ``frame_feats``
    ``static_feats``           ``static_feats``
    ``lengths`` / ``length``   ``lengths``
    ``delta_s``                ``delta_s``
    ``move_target``            ``move_target``
    ``process_label``          ``process_target``
    ``process_progress``       ``progress_target``
    ``trust_label``            ``trust_target``
    =========================  =========================

    Canonical keys (``pos``/``z``/``edge_index`` ...) pass through unchanged.
    Any other legacy key (``frame_mask``, ``move_mask``, ``process_mask``,
    ``q``, ``s`` ...) is preserved verbatim so no information is silently
    dropped (R5); :func:`validate_batch` ignores those passthrough keys.

    Returns a new ``dict``; the input is never mutated.
    """
    if isinstance(batch_dict, StandardBatch):
        batch_dict = batch_dict.as_mapping()
    if not isinstance(batch_dict, Mapping):
        raise ContractViolation(
            "BAD_BATCH_TYPE",
            f"to_standard expects a mapping, got {type(batch_dict).__name__!r}")
    _forbid_truth_fields(batch_dict)
    out: dict[str, Any] = {}
    passthrough: dict[str, Any] = {}
    for key, value in batch_dict.items():
        if not isinstance(key, str):
            raise ContractViolation(
                "BAD_BATCH_TYPE", f"batch keys must be str, got {type(key).__name__!r}")
        standard = LEGACY_TO_STANDARD.get(key, key)
        if standard in out and key in LEGACY_TO_STANDARD:
            # canonical/standard key already claimed: keep it, do not overwrite.
            passthrough[key] = value
            continue
        out[standard] = value
    out.update(passthrough)
    return out


def from_standard(batch: Any) -> dict[str, Any]:
    """Translate a standard batch back into legacy V5.6 trajectory key names.

    Inverse of :func:`to_standard` for the compat fields: ``process_target`` ->
    ``process_label``, ``progress_target`` -> ``process_progress``,
    ``trust_target`` -> ``trust_label``; the remaining legacy keys are identity
    mapped (batch-level ``lengths``).  Canonical keys pass through and unknown
    passthrough keys are preserved.

    Returns a new ``dict``; the input is never mutated.
    """
    if isinstance(batch, StandardBatch):
        batch = batch.as_mapping()
    if not isinstance(batch, Mapping):
        raise ContractViolation(
            "BAD_BATCH_TYPE",
            f"from_standard expects a StandardBatch or mapping, got "
            f"{type(batch).__name__!r}")
    _forbid_truth_fields(batch)
    out: dict[str, Any] = {}
    passthrough: dict[str, Any] = {}
    for key, value in batch.items():
        if not isinstance(key, str):
            raise ContractViolation(
                "BAD_BATCH_TYPE", f"batch keys must be str, got {type(key).__name__!r}")
        legacy = STANDARD_TO_LEGACY.get(key, key)
        if legacy in out and key in STANDARD_TO_LEGACY:
            passthrough[key] = value
            continue
        out[legacy] = value
    out.update(passthrough)
    return out
