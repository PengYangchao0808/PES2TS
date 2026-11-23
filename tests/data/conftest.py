"""Shared pytest setup for the Zone-2 (data engineering) tests.

Mirrors ``tests/conftest.py``: make the repo root importable no matter which
directory pytest is invoked from.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
