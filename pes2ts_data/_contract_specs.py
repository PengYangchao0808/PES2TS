"""Field specs and stable codes for the 3-layer data-batch contract (v2).

Internal module: the public API is :mod:`pes2ts_data.contracts`.  Kept separate
so ``contracts.py`` stays a thin, readable facade (project rule: files <= 350
lines).

Contract v2 layout
------------------
``BaseBatch``      identity/provenance layer: ``sample_id``, ``targets``,
                   ``split_tag``, ``provenance``.
``SequenceBatch``  legacy V5.6 trajectory compat fields (8).
``GraphBatch``     geometric batch fields (7).
``StandardBatch``  deprecated v1 union used only through the compatibility shim.

Constitution mapping: R1 (truth isolation), R5 (full-cost retention), R6 (split
firewall tags) and P2 (typed honest rejection with stable codes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

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
]

#: Current contract version (v2 = 3-layer model).  v1 was the flat
#: ``StandardBatch`` (kept as a deprecated shim).
CONTRACT_VERSION: Final[str] = "2.0.0"
#: Version of the deprecated flat batch (compat shim only).
LEGACY_CONTRACT_VERSION: Final[str] = "1.0.0"

#: dtype families accepted by :class:`FieldSpec`.
DTYPE_FAMILIES: Final[tuple[str, ...]] = (
    "float", "int", "bool", "str", "mapping", "any")

#: Stable machine-readable violation codes (append-only).
CODES: Final[tuple[str, ...]] = (
    "MISSING_FIELD",
    "BAD_DTYPE",
    "BAD_SHAPE",
    "BAD_VALUE",
    "BAD_INDEX",
    "BAD_BATCH_TYPE",
    "UNKNOWN_FIELD",
    "SHAPE_MISMATCH",
    "PROVENANCE_MISSING",
    "PROVENANCE_INVALID",
    "TARGETS_INVALID",
    "SPLIT_TAG_INVALID",
    "TRUTH_FIELD_FORBIDDEN",
)

#: Substrings that mark a key as ground-truth material (R1).  Matching is
#: case-insensitive on the normalized key (``-``/``.``/spaces -> ``_``).
TRUTH_FIELD_PATTERNS: Final[tuple[str, ...]] = (
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

#: Known split tags accepted by the batch contract (R6 vocabulary).
SPLIT_TAGS: Final[tuple[str, ...]] = (
    "train", "valid", "test", "id_dev", "scaffold_ood", "family_ood",
    "size_ood", "confirm", "unknown")


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
        Contract-level requirement (only ``pos``/``z``/``edge_index`` for the
        geometric layer); per-model requirements use ``REQUIRED_FIELDS``.
    description:
        One-line human description of the field semantics.
    provenance_sensitive:
        ``True`` when values derive from the audited teacher/truth path (R1):
        ``provenance`` is then mandatory.
    mutable:
        ``False`` for labels/targets/provenance that downstream code must treat
        as read-only.
    shape:
        Optional shape template.  ``int`` entries are enforced exactly;
        symbolic entries (``"N"`` nodes, ``"E"`` edges, ``"P"`` pairs) are used
        for cross-field consistency only.
    compat:
        ``True`` for legacy V5.6 trajectory compat fields.
    layer:
        ``"base"``, ``"graph"`` or ``"sequence"``.
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
    layer: str = "base"

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
        if self.layer not in ("base", "graph", "sequence"):
            raise ValueError(f"FieldSpec({self.name!r}): unknown layer {self.layer!r}")

    def accepts_ndim(self, ndim: int) -> bool:
        """Return ``True`` when ``ndim`` is an accepted rank."""
        ndims = self.ndim if isinstance(self.ndim, tuple) else (self.ndim,)
        return ndim in ndims


FIELD_SPECS: dict[str, FieldSpec] = {
    # -- base layer ---------------------------------------------------------
    "sample_id": FieldSpec(
        name="sample_id", dtype="str", ndim=0, required=False,
        description="Stable per-sample identity string (non-empty).",
        mutable=False, layer="base"),
    "targets": FieldSpec(
        name="targets", dtype="mapping", ndim=0, required=False,
        description="Mapping target-name -> target payload (may be empty).",
        mutable=False, layer="base"),
    "split_tag": FieldSpec(
        name="split_tag", dtype="str", ndim=0, required=False,
        description="Frozen split/eval-suite tag (R6 vocabulary).",
        mutable=False, layer="base"),
    "provenance": FieldSpec(
        name="provenance", dtype="any", ndim=0, required=False,
        description="Provenance token: Provenance, mapping with a token, or str.",
        provenance_sensitive=True, mutable=False, layer="base"),
    # -- graph layer --------------------------------------------------------
    "pos": FieldSpec(
        name="pos", dtype="float", ndim=2, required=True,
        description="(N,3) float atomic coordinates in Angstrom.",
        shape=("N", 3), layer="graph"),
    "z": FieldSpec(
        name="z", dtype="int", ndim=1, required=True,
        description="(N,) integer atomic numbers.", shape=("N",), layer="graph"),
    "edge_index": FieldSpec(
        name="edge_index", dtype="int", ndim=2, required=True,
        description="(2,E) integer edges [row=receiver, col=sender].",
        shape=(2, "E"), layer="graph"),
    "endpoint_mask": FieldSpec(
        name="endpoint_mask", dtype="bool", ndim=(1, 2), required=False,
        description="(N,) node-level or (B,) batch-level usable-endpoint mask.",
        shape=("N",), layer="graph"),
    "sym_pair_idx": FieldSpec(
        name="sym_pair_idx", dtype="int", ndim=2, required=False,
        description="(P,2) integer canonical symmetric key-atom pairs.",
        shape=("P", 2), layer="graph"),
    "target_xyz": FieldSpec(
        name="target_xyz", dtype="float", ndim=2, required=False,
        description="(N,3) float teacher-derived target geometry; provenance required (R1).",
        provenance_sensitive=True, mutable=False, shape=("N", 3), layer="graph"),
    "sample_weight": FieldSpec(
        name="sample_weight", dtype="float", ndim=1, required=False,
        description="(N,) float per-sample cost weight; full-cost retention (R5).",
        provenance_sensitive=True, mutable=False, shape=("N",), layer="graph"),
    # -- legacy V5.6 trajectory compat (sequence layer) ---------------------
    "frame_feats": FieldSpec(
        name="frame_feats", dtype="float", ndim=(2, 3), required=False,
        description="Compatibility: legacy frame features [T,F] / [B,T,F].",
        provenance_sensitive=True, compat=True, layer="sequence"),
    "static_feats": FieldSpec(
        name="static_feats", dtype="float", ndim=(1, 2), required=False,
        description="Compatibility: legacy static reaction features [S] / [B,S].",
        provenance_sensitive=True, compat=True, layer="sequence"),
    "lengths": FieldSpec(
        name="lengths", dtype="int", ndim=1, required=False,
        description="Compatibility: legacy per-trajectory lengths [B] (sample key 'length').",
        compat=True, layer="sequence"),
    "delta_s": FieldSpec(
        name="delta_s", dtype="float", ndim=(1, 2), required=False,
        description="Compatibility: legacy per-frame arc-length step delta_s.",
        compat=True, layer="sequence"),
    "move_target": FieldSpec(
        name="move_target", dtype="float", ndim=(1, 2, 3), required=False,
        description=(
            "Compatibility: MOVE target displacement. [T] scalar per-frame displacement"
            " when q_dim=1 (real idx_v2 data), [T,K] per-frame vectors, or [B,T,K]"
            " padded batches."),
        provenance_sensitive=True, mutable=False, compat=True, layer="sequence"),
    "process_target": FieldSpec(
        name="process_target", dtype="int", ndim=(1, 2), required=False,
        description="Compatibility: legacy PROCESS class labels (legacy 'process_label').",
        provenance_sensitive=True, mutable=False, compat=True, layer="sequence"),
    "progress_target": FieldSpec(
        name="progress_target", dtype="float", ndim=(1, 2), required=False,
        description="Compatibility: legacy PROCESS progress (legacy 'process_progress').",
        provenance_sensitive=True, mutable=False, compat=True, layer="sequence"),
    "trust_target": FieldSpec(
        name="trust_target", dtype="float", ndim=(1, 2), required=False,
        description="Compatibility: legacy trust target (legacy 'trust_label').",
        provenance_sensitive=True, mutable=False, compat=True, layer="sequence"),
}

#: Minimal geometric contract fields (enforced by default for graph batches).
CONTRACT_REQUIRED_FIELDS: Final[tuple[str, ...]] = ("pos", "z", "edge_index")
#: Base-layer fields.
BASE_FIELDS: Final[tuple[str, ...]] = ("sample_id", "targets", "split_tag", "provenance")
#: Identity fields required by default (``require_identity=True``).
IDENTITY_FIELDS: Final[tuple[str, ...]] = ("sample_id", "provenance")
#: Graph-layer fields.
GRAPH_FIELDS: Final[tuple[str, ...]] = (
    "pos", "z", "edge_index", "endpoint_mask", "sym_pair_idx", "target_xyz",
    "sample_weight")
#: Sequence-layer compat fields.
COMPAT_FIELDS: Final[tuple[str, ...]] = tuple(
    name for name, spec in FIELD_SPECS.items() if spec.compat)

#: Canonical compat field -> legacy V5.6 trajectory key.
LEGACY_FIELD_MAP: Final[dict[str, str]] = {
    "frame_feats": "frame_feats",
    "static_feats": "static_feats",
    "lengths": "lengths",           # single-sample legacy key is "length"
    "delta_s": "delta_s",
    "move_target": "move_target",
    "process_target": "process_label",
    "progress_target": "process_progress",
    "trust_target": "trust_label",
}
LEGACY_TO_STANDARD: Final[dict[str, str]] = {
    **{legacy: standard for standard, legacy in LEGACY_FIELD_MAP.items()},
    "length": "lengths",
    "provenance_id": "provenance",
}
STANDARD_TO_LEGACY: Final[dict[str, str]] = dict(LEGACY_FIELD_MAP)
