"""Capture the PRE-R1 golden reference for the A0-equivalence gate.

Run this BEFORE modifying pes2ts_core/pesn/offline/training/key_atom_bc.py:
it loads tests/pesn/test_key_atom_trainer_r1.py by path, executes the
frozen fixture + call shapes, and dumps JSON that gets pasted into
``GOLDEN_PRECHANGE``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] if Path(__file__).is_absolute() else Path.cwd()
REPO = Path("/mnt/e/Calculations/AI4S_ML_Studys/PES2TS")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "r1test", REPO / "tests/pesn/test_key_atom_trainer_r1.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _jsonable(out):
    row = {}
    for k, v in out.items():
        if hasattr(v, "item") and not isinstance(v, (int, float, str)):
            v = v.item()
        if isinstance(v, bool):
            row[k] = v
        elif isinstance(v, (int, float, str)) or v is None:
            row[k] = v
        else:  # numpy scalar
            row[k] = v.item()
    return row


inputs = mod._golden_inputs()
calls = mod._golden_calls(inputs)
golden = {name: _jsonable(out) for name, out in calls.items()}
print(json.dumps(golden, indent=1, sort_keys=True, ensure_ascii=False))
