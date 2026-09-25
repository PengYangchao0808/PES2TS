"""Atomic Parquet IO with deterministic column ordering.

Columns are always written sorted by name, so re-materializing the same data
yields a stable column order.  Writes reuse the shared ``atomic_writer``
primitive from :mod:`pes2ts_core.utils.jsonio`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from pes2ts_core.utils.hashing import JSONValue
from pes2ts_core.utils.jsonio import atomic_writer

if TYPE_CHECKING:
    import pandas as pd

#: One column of values accepted in a mapping input.
type ParquetColumn = list[JSONValue] | pa.Array | pa.ChunkedArray
#: Accepted inputs: a pyarrow table, a column-name -> values mapping, or a
#: pandas DataFrame (converted through pyarrow; pandas semantics are not used).
type ParquetInput = pa.Table | dict[str, ParquetColumn] | pd.DataFrame


def _to_arrow_table(table: ParquetInput) -> pa.Table:
    """Return *table* as a pyarrow table, converting mappings/DataFrames."""
    if isinstance(table, pa.Table):
        return table
    return pa.table(table)


def write_parquet(path: str | Path, table: ParquetInput) -> None:
    """Atomically write *table* to *path* with columns sorted by name.

    The parent directory is created when missing; on failure no temp file is
    left behind and any pre-existing *path* stays untouched.
    """
    logger = logging.getLogger(__name__)
    arrow_table = _to_arrow_table(table)
    ordered = arrow_table.select(sorted(arrow_table.column_names))
    with atomic_writer(path) as handle:
        pq.write_table(ordered, handle)
    logger.debug(
        "Wrote parquet %s (%d rows, %d columns)",
        path,
        ordered.num_rows,
        ordered.num_columns,
    )


def read_parquet(path: str | Path) -> pa.Table:
    """Read the Parquet table at *path* as a pyarrow table."""
    return pq.read_table(path)
