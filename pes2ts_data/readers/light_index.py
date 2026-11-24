"""Read-only port of the V5.3-MP light-index pickle reader (Zone 2).

Source mapping (ported from the frozen ``pes2ts_core/pesn`` tree, removed from
master): ``key_atom_dataset_v3.py`` — ``LIGHT_CACHE_PKL`` /
``LIGHT_CACHE_JSON`` / ``LIGHT_CACHE_SCHEMA`` and the ``light_cache=True``
read path of :class:`ShardedKeyAtomDatasetV3` (binding semantics preserved:
the cache is never served when the data-manifest / split-manifest / d0 / filter
binding drifts).

The pickle stores instances of classes from the removed frozen tree.  A
restricted unpickler maps only those legacy class paths to a stand-in object
with the same attribute surface and refuses every other module, so the frozen
artifact stays readable after the source modules are deleted.  Read-only: no
writes, no torch, no truth access.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from ._common import ReaderError, require

__all__ = [
    "LIGHT_CACHE_JSON",
    "LIGHT_CACHE_PKL",
    "LIGHT_CACHE_SCHEMA",
    "LIGHT_INDEX_KEYS",
    "LightIndexState",
    "light_index_available",
    "light_index_binding",
    "load_light_index",
    "read_light_index_metadata",
]

#: Light-index filenames + schema id (frozen ``v53mp_light_index_v1``).
LIGHT_CACHE_PKL: Final[str] = ".light_index_v53mp.pkl"
LIGHT_CACHE_JSON: Final[str] = ".light_index_v53mp.json"
LIGHT_CACHE_SCHEMA: Final[str] = "v53mp_light_index_v1"

#: State keys of the light-index pickle (frozen ``_try_load_light_cache``).
LIGHT_INDEX_KEYS: Final[tuple[str, ...]] = (
    "node_index", "curve_atomic", "curve_energy", "samples", "weight_totals")

#: Legacy package roots whose pickled classes are mapped to the stand-in.
_LEGACY_ROOTS: Final[tuple[str, ...]] = ("pes2ts_core", "pes2ts_v5")

#: Stdlib/numpy modules that may legitimately appear inside the frozen pickle.
_SAFE_PICKLE_MODULES: Final[frozenset[str]] = frozenset({
    "builtins", "collections", "collections.abc", "copyreg", "types",
    "numpy", "numpy.core.multiarray", "numpy._core.multiarray",
    "numpy.core.numeric", "numpy.dtype",
})


class _LegacyStandIn:
    """Stand-in for pickled classes whose defining modules were removed."""

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(dict(state))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LegacyStandIn({sorted(self.__dict__)})"


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that refuses arbitrary classes and shims the removed ones."""

    def find_class(self, module: str, name: str) -> Any:
        if module.split(".")[0] in _LEGACY_ROOTS:
            return _LegacyStandIn
        if module in _SAFE_PICKLE_MODULES or module.startswith("numpy"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"refusing to unpickle class {module}.{name}: not part of the "
            "frozen light-index schema")


@dataclass(frozen=True)
class LightIndexState:
    """Read-only contents of ``.light_index_v53mp.pkl`` + its binding json."""

    node_index: dict[str, set[int]]
    curve_atomic: dict[str, tuple[int, ...]]
    curve_energy: dict[str, dict[int, float]]
    samples: tuple[Any, ...]
    weight_totals: dict[str, float]
    metadata: dict[str, Any]
    tables_dir: Path

    @property
    def n_samples(self) -> int:
        return len(self.samples)


def light_index_available(tables_dir: str | Path) -> bool:
    """Return ``True`` when both light-index artifacts exist."""
    tables_dir = Path(tables_dir)
    return ((tables_dir / LIGHT_CACHE_PKL).is_file()
            and (tables_dir / LIGHT_CACHE_JSON).is_file())


def read_light_index_metadata(tables_dir: str | Path) -> dict[str, Any]:
    """Read only the light-index binding json (never unpickles)."""
    path = Path(tables_dir) / LIGHT_CACHE_JSON
    require(path.is_file(), "LIGHT_INDEX_MISSING", f"{path} not found")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReaderError("LIGHT_INDEX_PARSE", f"{path}: {exc}") from exc
    require(isinstance(doc.get("binding"), dict), "LIGHT_INDEX_SCHEMA",
            f"{path}: 'binding' mapping missing")
    return doc


def light_index_binding(
    *,
    data_manifest_sha256: str,
    split_manifest_sha256: str,
    d0: float,
    label_statuses: Sequence[str] | set[str],
    directions: Sequence[str] | set[str],
    fold: int | None,
) -> dict[str, Any]:
    """Rebuild the frozen content binding for a light-index verification."""
    return {
        "schema": LIGHT_CACHE_SCHEMA,
        "data_manifest_sha256": data_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "d0": float(d0),
        "label_statuses": sorted(label_statuses),
        "directions": sorted(directions),
        "fold": fold,
    }


def load_light_index(
    tables_dir: str | Path,
    *,
    expected_binding: Mapping[str, Any] | None = None,
) -> LightIndexState:
    """Load the light-index pickle (fail-closed, restricted unpickler).

    ``expected_binding`` (see :func:`light_index_binding`) makes the load
    fail-closed on drift (``LIGHT_INDEX_STALE``); when omitted the binding json
    is parsed but not compared.
    """
    tables_dir = Path(tables_dir)
    json_path = tables_dir / LIGHT_CACHE_JSON
    pkl_path = tables_dir / LIGHT_CACHE_PKL
    require(json_path.is_file() and pkl_path.is_file(), "LIGHT_INDEX_MISSING",
            f"{tables_dir}: expected {LIGHT_CACHE_PKL} and {LIGHT_CACHE_JSON}")
    doc = read_light_index_metadata(tables_dir)
    binding = doc["binding"]
    require(binding.get("schema") == LIGHT_CACHE_SCHEMA, "LIGHT_INDEX_SCHEMA",
            f"{json_path}: binding schema {binding.get('schema')!r}")
    if expected_binding is not None:
        require(dict(binding) == dict(expected_binding), "LIGHT_INDEX_STALE",
                "light index binding differs from the expected "
                "manifest/split/d0/filter binding")
    try:
        with pkl_path.open("rb") as handle:
            state = _RestrictedUnpickler(handle).load()
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError,
            ImportError, ValueError) as exc:
        raise ReaderError("LIGHT_INDEX_UNREADABLE", f"{pkl_path}: {exc}") from exc
    require(isinstance(state, Mapping), "LIGHT_INDEX_SHAPE",
            f"{pkl_path}: state is not a mapping")
    missing = [key for key in LIGHT_INDEX_KEYS if key not in state]
    require(not missing, "LIGHT_INDEX_SHAPE",
            f"{pkl_path}: missing state keys {missing}")
    return LightIndexState(
        node_index={str(k): set(v) for k, v in dict(state["node_index"]).items()},
        curve_atomic={str(k): tuple(v) for k, v in dict(state["curve_atomic"]).items()},
        curve_energy={str(k): {int(n): float(e) for n, e in dict(v).items()}
                      for k, v in dict(state["curve_energy"]).items()},
        samples=tuple(state["samples"]),
        weight_totals={str(k): float(v) for k, v in
                       dict(state["weight_totals"]).items()},
        metadata=doc,
        tables_dir=tables_dir,
    )
