"""V5.5 dataset manifest and the stable internal interface ``V55DatasetSpec``.

Design note (contract timing)
-----------------------------
The V5.5 data/label contract document
(``docs/plans/PES2TS_V5.5_数据与标签契约_v1_*.md``) is authored in parallel.
Until it lands, V5.6 defines this **stable internal interface** itself so that
models/training/evaluation can be built and tested against a synthetic fixture.
When the contract lands, :func:`adapt_v55_manifest` maps it onto
:class:`V55DatasetSpec`; downstream modules only ever see this interface.

Artifact layout (directory):

    manifest.json         # this module's manifest (spec + counts + hashes)
    reactions.npz         # reaction-level metadata + static descriptor
    trajectories.npz      # trajectory-level index/length/direction
    frames.npz            # frame-level tensors (features, labels, masks, q)
    geom.npz              # kinematic geometry basis (R1 rollout only, optional)
    splits.json           # split name -> reaction_id list
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "pes2ts_v5_dataset_v1"
MANIFEST_NAME = "manifest.json"
ARTIFACT_FILES = (
    "reactions.npz",
    "trajectories.npz",
    "frames.npz",
    "geom.npz",
    "splits.json",
)
SPLIT_NAMES = ("train", "valid", "id", "scaffold_ood", "family_ood", "size_ood")


def sha256_file(path: str | Path) -> str:
    """SHA256 of a file (chunked, no external dependency)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_dataset_hash(data_dir: str | Path, files: tuple[str, ...] | None = None) -> str:
    """Content hash over artifact files: ``name:sha256`` lines, sorted by name."""
    root = Path(data_dir)
    names = files or ARTIFACT_FILES
    lines = []
    for name in sorted(names):
        path = root / name
        if path.exists():
            lines.append(f"{name}:{sha256_file(path)}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


class V55ContractNotAvailable(RuntimeError):
    """Raised when a V5.5 contract artifact is required but not found."""


@dataclass(frozen=True)
class V55DatasetSpec:
    """Stable internal V5.5-side data interface used by all V5.6 code.

    Field semantics mirror the research plan v2.1 §2 / 附录 B.1–B.2:
    four-level counts, geometry-only (G0) vs cached-QC (G1), trajectory-level
    sequences, and the frozen label/mask contract.
    """

    schema_version: str = SCHEMA_VERSION
    dataset_version: str = "synthetic-v1"
    # input mode: "G0" geometry-only primary screen; "G1" cached-QC secondary
    input_mode: str = "G0"
    # dimensions
    frame_feat_dim: int = 11
    static_feat_dim: int = 8
    q_dim: int = 3
    atom_feat_dim: int = 0
    # sequence limits
    history_len: int = 16
    max_trajectory_len: int = 64
    # label contract
    process_classes: int = 3
    near_ts_window: float = 0.10
    trust_definition: str = "curvature_anomaly_proxy"
    # splits (reaction_id level group split)
    splits: tuple[str, ...] = SPLIT_NAMES
    # provenance of the upstream contract when available (None = built-in spec)
    contract_doc: str | None = None
    contract_sha256: str | None = None
    notes: str = "stable internal spec; synthetic fixture until V5.5 contract lands"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["splits"] = list(self.splits)
        return d

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "V55DatasetSpec":
        known = {f for f in cls.__dataclass_fields__ if f != "splits"}
        kwargs = {k: doc[k] for k in doc if k in known}
        if "splits" in doc:
            kwargs["splits"] = tuple(doc["splits"])
        return cls(**kwargs)


@dataclass
class DatasetManifest:
    """Manifest document for a V5.5-shaped dataset artifact directory."""

    spec: V55DatasetSpec
    dataset_version: str
    dataset_hash: str
    files: dict[str, dict] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    splits_counts: dict[str, int] = field(default_factory=dict)
    normalization: dict[str, Any] = field(default_factory=dict)
    label_spec: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "spec": self.spec.to_dict(),
            "dataset_version": self.dataset_version,
            "dataset_hash": self.dataset_hash,
            "files": self.files,
            "counts": self.counts,
            "splits_counts": self.splits_counts,
            "normalization": self.normalization,
            "label_spec": self.label_spec,
            "created_at": self.created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notes": self.notes,
        }

    def save(self, data_dir: str | Path) -> Path:
        root = Path(data_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / MANIFEST_NAME
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, data_dir: str | Path) -> "DatasetManifest":
        root = Path(data_dir)
        path = root / MANIFEST_NAME
        if not path.exists():
            raise V55ContractNotAvailable(f"dataset manifest missing: {path}")
        doc = json.loads(path.read_text(encoding="utf-8"))
        schema = doc.get("schema_version")
        if schema != SCHEMA_VERSION:
            raise V55ContractNotAvailable(
                f"unsupported dataset schema {schema!r}; expected {SCHEMA_VERSION!r}")
        return cls(
            spec=V55DatasetSpec.from_dict(doc.get("spec", {})),
            dataset_version=doc.get("dataset_version", "unknown"),
            dataset_hash=doc.get("dataset_hash", ""),
            files=doc.get("files", {}),
            counts=doc.get("counts", {}),
            splits_counts=doc.get("splits_counts", {}),
            normalization=doc.get("normalization", {}),
            label_spec=doc.get("label_spec", {}),
            created_at=doc.get("created_at", ""),
            notes=doc.get("notes", ""),
        )

    def verify(self, data_dir: str | Path) -> dict:
        """Verify artifact SHA256 values; returns a report dict (never raises)."""
        root = Path(data_dir)
        ok, missing, mismatch = [], [], []
        for name, rec in sorted(self.files.items()):
            path = root / name
            if not path.exists():
                missing.append(name)
                continue
            actual = sha256_file(path)
            if rec.get("sha256") == actual:
                ok.append(name)
            else:
                mismatch.append(name)
        recomputed = compute_dataset_hash(root)
        return {
            "ok": ok,
            "missing": missing,
            "mismatch": mismatch,
            "dataset_hash_expected": self.dataset_hash,
            "dataset_hash_actual": recomputed,
            "hash_matches": recomputed == self.dataset_hash,
        }


def adapt_v55_manifest(contract_path: str | Path, data_dir: str | Path) -> V55DatasetSpec:
    """Adapter hook: map the上游 V5.5 contract manifest onto ``V55DatasetSpec``.

    Raises :class:`V55ContractNotAvailable` while the V5.5 contract/adapter is
    not present.  Downstream code must call this only when
    ``manifest.spec.contract_doc`` is set; synthetic fixtures use the built-in
    spec directly.
    """
    path = Path(contract_path)
    if not path.exists():
        raise V55ContractNotAvailable(
            "V5.5 data/label contract not available yet; continue with the "
            f"built-in V55DatasetSpec + synthetic fixture (looked for {path})")
    doc = json.loads(path.read_text(encoding="utf-8"))
    raw_spec = doc.get("spec") or doc.get("v55_spec") or {}
    if not raw_spec:
        raise V55ContractNotAvailable(
            f"{path} does not expose a 'spec' mapping; adapter must be updated")
    spec = V55DatasetSpec.from_dict(raw_spec)
    return V55DatasetSpec(
        **{**spec.to_dict(), "splits": tuple(spec.splits),
           "contract_doc": str(path), "contract_sha256": sha256_file(path)})
