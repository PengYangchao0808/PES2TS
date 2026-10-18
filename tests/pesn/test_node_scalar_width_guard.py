"""v4.1 WP1 acceptance addendum (finding 4): node-scalar width guard.

The load-time guard ``NODE_SCALAR_WIDTH_MISMATCH`` (``dual_egnn.py``
``state_inputs_to_tensors``) is the loud-failure gate that makes old
7-channel v2 checkpoints/shards intentionally incompatible with the new
5-channel v3 model.  A silent removal of this guard during WP2/WP4
refactoring would feed a wrong-shaped tensor to ``node_proj`` — this test
pins it three ways:

  (a) a synthetic ``(N,7)`` state is rejected by both
      ``state_inputs_to_tensors`` and ``collate_dual_states``;
  (b) a REAL 7-channel v2 shard row is rejected with the same typed error;
  (c) a REAL 5-channel v41 shard row passes (collates to width 5).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pes2ts_core.pesn.features.dual_ts_graph import (  # noqa: E402
    DualTSGraphError,
)
from pes2ts_core.pesn.models.dual_egnn import (  # noqa: E402
    collate_dual_states,
    state_inputs_to_tensors,
)

BASE = ROOT / "outputs/pesn/dual_ts_v2_20261015"
OLD_SHARD = BASE / "wp3_features/features_RXN_0000000160.npz"
V41_SHARD = BASE / "wp3_features_v41/features_RXN_0000000160.npz"

Z3 = np.asarray([6, 6, 1], dtype=int)  # C, C, H


def _synthetic_state(n_scalar_channels: int) -> dict:
    """A minimal valid shard-row state dict with a chosen node_scalar width."""
    rng = np.random.default_rng(0)
    n, pairs = 3, np.array([[0, 1], [0, 2]])
    directed = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T
    return {
        "atomic_numbers": Z3.copy(),
        "node_scalars": rng.normal(size=(n, n_scalar_channels)),
        "x_angstrom": rng.normal(size=(n, 3)),
        "g_scaled": rng.normal(size=(n, 3)),
        "atom_feats_a": rng.random((n, 4)),
        "atom_feats_b": rng.random((n, 4)),
        "edge_feats_a": rng.random((2 * len(pairs), 4)),
        "edge_feats_b": rng.random((2 * len(pairs), 4)),
        "edge_index": directed,
    }


def _state_from_shard_row(shard, row: int = 0) -> dict:
    """One REAL shard row as a collatable state dict (shared topology blocks,
    per-state scalars/vectors)."""
    return {
        "atomic_numbers": shard["atomic_numbers"],
        "node_scalars": shard["node_scalars"][row],
        "x_angstrom": shard["x_angstrom"][row],
        "g_scaled": shard["g_scaled"][row],
        "atom_feats_a": shard["atom_feats_a"],
        "atom_feats_b": shard["atom_feats_b"],
        "edge_feats_a": shard["edge_feats_a"],
        "edge_feats_b": shard["edge_feats_b"],
        "edge_index": shard["edge_index"],
    }


def _assert_width_mismatch(exc: pytest.ExceptionInfo) -> None:
    assert isinstance(exc.value, DualTSGraphError)
    assert "NODE_SCALAR_WIDTH_MISMATCH" in str(exc.value)


def test_synthetic_7_channel_state_rejected():
    bad = _synthetic_state(7)
    with pytest.raises(DualTSGraphError) as ei:
        state_inputs_to_tensors(torch, bad)
    _assert_width_mismatch(ei)
    with pytest.raises(DualTSGraphError) as ei:
        collate_dual_states(torch, [bad])
    _assert_width_mismatch(ei)


def test_real_v2_7_channel_shard_row_rejected():
    if not OLD_SHARD.is_file():
        pytest.skip("historical v2 7-channel shard not present")
    with np.load(OLD_SHARD, allow_pickle=False) as shard:
        assert shard["node_scalars"].shape[-1] == 7
        row = _state_from_shard_row(shard, 0)
    with pytest.raises(DualTSGraphError) as ei:
        collate_dual_states(torch, [row])
    _assert_width_mismatch(ei)


def test_real_v41_5_channel_shard_row_passes():
    if not V41_SHARD.is_file():
        pytest.skip("v41 5-channel shard not present")
    with np.load(V41_SHARD, allow_pickle=False) as shard:
        assert shard["node_scalars"].shape[-1] == 5
        n_atoms = int(shard["node_scalars"].shape[1])
        row = _state_from_shard_row(shard, 0)
    batch = collate_dual_states(torch, [row])
    assert int(batch["node_scalars"].shape[-1]) == 5
    assert int(batch["node_scalars"].shape[0]) == n_atoms
