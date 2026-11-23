"""Legacy tier: historical V4/V5.1-V5.4 tests (skipped by default).

Every module under ``tests/legacy/`` is auto-marked ``legacy`` at collection
time so ``pytest -q`` keeps the daily gate fast; the CI full tier runs
``pytest -q -m "not (realdata or xtb or acp or orca)"`` and therefore
still exercises them.
"""

from __future__ import annotations

import pytest


def pytest_itemcollected(item: pytest.Item) -> None:
    """Mark every collected legacy-tier item before marker de-selection."""
    item.add_marker(pytest.mark.legacy)
