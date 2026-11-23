"""Unbounded dual-TS displacement head tests (synthetic only, CPU, float64).

Covers the contract §5 / ADR-0024 item 1-2 guarantees:

* ``a_hat = readout(h, v)`` is UNBOUNDED — a large input produces a large
  output (no ``0.1·tanh`` clamp, no inverse-Jacobian);
* the H1 frozen prior ``a_phys`` is a plain forward argument, never a
  registered parameter;
* ``residual_norm`` / ``skip_norm`` (and the zero-denominator ``NaN`` ratio)
  are reported per atom;
* batch assembly (concatenated-graph) yields one ``(N,3)`` action over the
  full batch;
* the encoder is duck-typed: ``(h, v)`` tuple and ``(N, C, 3)`` vector
  channels are both accepted.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pes2ts_core.pesn.models import require_torch
from pes2ts_core.pesn.models.dual_egnn import (
    DualTSModelConfig,
    collate_dual_states,
)
from pes2ts_core.pesn.models.dual_egnn_painn import DualTSPaiNNEncoderConfig
from pes2ts_core.pesn.models.dual_ts_displacement import (
    DUAL_TS_UNBOUNDED_DISPLACEMENT_VERSION,
    DisplacementHeadConfig,
    DualTSDisplacementPolicy,
    build_displacement_policy,
)

torch = require_torch()
TOL = 1e-6


def synthetic_states(n_states: int = 2, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = 6
    pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 3], [1, 4]])
    directed = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T
    states = []
    for _ in range(n_states):
        x = rng.normal(size=(n, 3))
        x -= x.mean(axis=0)
        states.append({
            "atomic_numbers": np.array([6, 1, 7, 6, 8, 1]),
            "node_scalars": rng.normal(size=(n, 5)),
            "x_angstrom": x,
            "g_scaled": rng.normal(size=(n, 3)),
            "atom_feats_a": rng.random((n, 4)),
            "atom_feats_b": rng.random((n, 4)),
            "edge_feats_a": rng.random((2 * len(pairs), 4)),
            "edge_feats_b": rng.random((2 * len(pairs), 4)),
            "edge_index": directed,
        })
    return states


def make_policy(use_prior_skip: bool = False, kind: str = "egnn",
                hidden: int = 32, n_layers: int = 2) -> DualTSDisplacementPolicy:
    torch.manual_seed(11)
    pol = build_displacement_policy(
        DualTSModelConfig(),
        DisplacementHeadConfig(readout_hidden=16, use_prior_skip=use_prior_skip),
        encoder_kind=kind, hidden=hidden, n_layers=n_layers,
    ).to(torch.float64)
    pol.train(False)
    return pol


# ---------------------------------------------------------------------------
# (1) unbounded output
# ---------------------------------------------------------------------------

def test_a_hat_is_unbounded_large_input_large_output():
    pol = make_policy(use_prior_skip=False)
    batch = collate_dual_states(torch, synthetic_states())
    with torch.no_grad():
        base = pol(batch)["a_hat"]
        big = dict(batch)
        big["g"] = batch["g"] * 100.0
        out = pol(big)["a_hat"]
    base_norm = base.norm(dim=-1).mean()
    out_norm = out.norm(dim=-1).mean()
    # the output scales with the gradient magnitude — no 0.1 Å ceiling.
    assert float(out_norm) > 10.0 * float(base_norm)
    assert float(out.norm(dim=-1).max()) > 0.1


def test_version_string():
    assert DUAL_TS_UNBOUNDED_DISPLACEMENT_VERSION == "dual_ts_unbounded_displacement_v1"


# ---------------------------------------------------------------------------
# (2) H1 frozen prior is NOT a parameter
# ---------------------------------------------------------------------------

def test_a_phys_skip_not_in_parameters():
    pol = make_policy(use_prior_skip=True)
    names = [n for n, _ in pol.named_parameters()]
    assert not any(("a_phys" in n) or ("prior" in n) or ("skip" in n) for n in names)
    # parameter count == encoder + readout head only (the prior contributes 0).
    expected = (sum(p.numel() for p in pol.encoder.parameters())
                + sum(p.numel() for p in pol.readout_mlp.parameters()))
    assert pol.n_parameters == expected


def test_missing_a_phys_raises_when_skip_enabled():
    pol = make_policy(use_prior_skip=True)
    batch = collate_dual_states(torch, synthetic_states())
    with pytest.raises(ValueError, match="a_phys"):
        pol(batch)


# ---------------------------------------------------------------------------
# (3) residual / skip norms and the zero-denominator NaN ratio
# ---------------------------------------------------------------------------

def test_residual_skip_norms_and_nan_ratio():
    pol = make_policy(use_prior_skip=True)
    batch = collate_dual_states(torch, synthetic_states())
    n = int(batch["z"].shape[0])
    a_phys = torch.zeros(n, 3, dtype=torch.float64)
    a_phys[0] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    a_phys[1] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)  # zero skip
    with torch.no_grad():
        out = pol(batch, a_phys=a_phys)
    # a_hat = a_phys + residual, exactly.
    assert torch.allclose(out["a_hat"], a_phys + out["residual"], atol=TOL)
    assert torch.allclose(out["residual_norm"], out["residual"].norm(dim=-1), atol=TOL)
    assert torch.allclose(out["skip_norm"], a_phys.norm(dim=-1), atol=TOL)
    # zero-denominator ratio is NaN (single column, never inf).
    assert torch.isnan(out["residual_skip_ratio"][1]).item()
    live = out["skip_norm"] > 0.0
    assert torch.allclose(
        out["residual_skip_ratio"][live],
        out["residual_norm"][live] / out["skip_norm"][live], atol=TOL)


# ---------------------------------------------------------------------------
# (4) batch assembly
# ---------------------------------------------------------------------------

def test_batch_assembly_shape():
    pol = make_policy(use_prior_skip=False)
    states = synthetic_states(3)
    batch = collate_dual_states(torch, states)
    n_total = sum(int(np.asarray(s["atomic_numbers"]).size) for s in states)
    with torch.no_grad():
        out = pol(batch)
    assert int(batch["n_graphs"]) == 3
    assert out["a_hat"].shape == (n_total, 3)
    assert out["h"].shape == (n_total, pol.config.hidden)
    assert out["v"].shape == (n_total, 3)


# ---------------------------------------------------------------------------
# (5) duck-typed encoder: tuple return + (N, C, 3) channels
# ---------------------------------------------------------------------------

class _DummyEncoder(torch.nn.Module):
    def __init__(self, hidden: int, C: int):
        super().__init__()
        self.hidden = hidden
        self.C = C
        self.lin = torch.nn.Linear(5, hidden)

    def forward(self, inputs):
        n = int(inputs["z"].shape[0])
        h = self.lin(inputs["node_scalars"])
        v = torch.ones(n, self.C, 3, device=h.device, dtype=h.dtype)
        return h, v  # a TUPLE, not a dict


def test_duck_typed_tuple_and_multichannel_readout():
    torch.manual_seed(3)
    cfg = DualTSModelConfig(hidden=8)
    dummy = _DummyEncoder(hidden=8, C=3).to(torch.float64)
    pol = DualTSDisplacementPolicy(
        dummy, cfg, DisplacementHeadConfig(readout_hidden=8)).to(torch.float64)
    batch = collate_dual_states(torch, synthetic_states(2))
    with torch.no_grad():
        out = pol(batch)
    n = int(batch["z"].shape[0])
    assert out["a_hat"].shape == (n, 3)
    # the (N, C, 3) readout is a shared scalar gate across the C channels.
    gate = pol.readout_mlp(dummy.lin(batch["node_scalars"]))
    expected = (gate.unsqueeze(-1) * torch.ones(n, 3, 3, dtype=torch.float64)).sum(1)
    assert torch.allclose(out["a_hat"], expected, atol=TOL)


def test_painn_encoder_path_builds():
    pol = make_policy(use_prior_skip=False, kind="painn", hidden=32, n_layers=2)
    batch = collate_dual_states(torch, synthetic_states())
    with torch.no_grad():
        out = pol(batch)
    assert out["a_hat"].shape == (int(batch["z"].shape[0]), 3)
    assert isinstance(pol.config, DualTSPaiNNEncoderConfig)
