"""Zone-2 data readers (pure numpy/pyarrow, no torch, read-only).

Ported from the frozen ``pes2ts_core/pesn`` tree and ``pes2ts_v5/data`` during
the repo restructuring so the datastore stays readable after those modules are
removed from master:

* :mod:`.idx_v2` — memmap reader over the frozen ``v53mp_idx_v2`` trajectory
  index (``curves.json`` + per-group ``.npy`` arrays; read-only).
* :mod:`.idx_v2_dataset` — V5.5-shaped numpy dataset view over that index
  (scene features, MOVE/PROCESS/TRUST targets; no torch collation).
* :mod:`.key_atom_stream` — streaming read-only view over the six frozen CSV
  tables (curve ids, node windows, survival-filtered sample columns, label
  joins); never materialises the 14.7M-row sample list.
* :mod:`.label_cache` — read-only ``labels_cache_v53mp.npz`` label-cache loader
  and the restricted ``.light_index_v53mp.pkl`` light-index loader.

Stable public API::

    from pes2ts_data.readers import (
        READER_VERSION, ReaderError,
        IdxV2Reader, idx_v2_available, IdxV2Error, read_staging_chunk,
        IdxV2Dataset, IdxV2Spec, build_frame_features,
        KeyAtomStreamView, StreamColumns, IntentRow,
        load_label_cache, LabelTables, load_light_index, LightIndexState,
        light_index_available, light_index_binding, read_light_index_metadata,
    )
"""

from __future__ import annotations

from ._common import (
    D0_PRIMARY_ANGSTROM,
    DATA_MANIFEST_FILENAME,
    LABEL_STATUSES,
    REQUIRED_COLUMNS,
    SPLIT_MANIFEST_FILENAME,
    TABLE_FILENAMES,
    WEIGHT_SUM_TOL,
    ReaderError,
    check_columns,
    load_folds,
    read_csv_rows,
    sha256_file,
    verify_data_manifest,
)
from .idx_v2 import (
    DEFAULT_IDX_ROOT,
    DEFAULT_STAGING_ROOT,
    SCHEMA,
    STATUS_CONFLICT,
    STATUS_MASKED,
    STATUS_OK,
    STATUS_TERMINAL,
    IdxV2Error,
    IdxV2Reader,
    idx_v2_available,
    read_staging_chunk,
)
from .idx_v2_dataset import IdxV2Dataset
from .idx_v2_features import IdxV2Spec, build_frame_features
from .key_atom_stream import IntentRow, KeyAtomStreamView, StreamColumns
from .label_cache import (
    CACHE_JSON,
    CACHE_NPZ,
    CACHE_SCHEMA,
    CurveLabels,
    EndpointSampleLabel,
    LabelTables,
    V52SampleLabel,
    load_label_cache,
)
from .light_index import (
    LIGHT_CACHE_JSON,
    LIGHT_CACHE_PKL,
    LIGHT_CACHE_SCHEMA,
    LightIndexState,
    light_index_available,
    light_index_binding,
    load_light_index,
    read_light_index_metadata,
)

__version__ = "1.0.0"
READER_VERSION = __version__

__all__ = [
    "__version__",
    "READER_VERSION",
    # -- common ------------------------------------------------------------
    "D0_PRIMARY_ANGSTROM",
    "DATA_MANIFEST_FILENAME",
    "LABEL_STATUSES",
    "REQUIRED_COLUMNS",
    "SPLIT_MANIFEST_FILENAME",
    "TABLE_FILENAMES",
    "WEIGHT_SUM_TOL",
    "ReaderError",
    "check_columns",
    "load_folds",
    "read_csv_rows",
    "sha256_file",
    "verify_data_manifest",
    # -- idx_v2 ------------------------------------------------------------
    "DEFAULT_IDX_ROOT",
    "DEFAULT_STAGING_ROOT",
    "SCHEMA",
    "STATUS_CONFLICT",
    "STATUS_MASKED",
    "STATUS_OK",
    "STATUS_TERMINAL",
    "IdxV2Error",
    "IdxV2Reader",
    "idx_v2_available",
    "read_staging_chunk",
    "IdxV2Dataset",
    "IdxV2Spec",
    "build_frame_features",
    # -- six-table stream --------------------------------------------------
    "IntentRow",
    "KeyAtomStreamView",
    "StreamColumns",
    # -- label cache / light index ----------------------------------------
    "CACHE_JSON",
    "CACHE_NPZ",
    "CACHE_SCHEMA",
    "LIGHT_CACHE_JSON",
    "LIGHT_CACHE_PKL",
    "LIGHT_CACHE_SCHEMA",
    "CurveLabels",
    "EndpointSampleLabel",
    "LabelTables",
    "LightIndexState",
    "V52SampleLabel",
    "light_index_available",
    "light_index_binding",
    "load_label_cache",
    "load_light_index",
    "read_light_index_metadata",
]
