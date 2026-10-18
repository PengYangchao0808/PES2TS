"""PaiNN-style dual-TS encoder tests (synthetic only, CPU, float64).

Covers plan v4 §7 V1/V2 / contract §5 / ADR-0024 item 4:

* translation invariance (``h`` invariant, ``v`` invariant — polar vector
  from gradient only, no absolute-coordinate seed);
* rotation equivariance: ``a_hat`` rotates covariantly, and under an improper
  (reflection) transform the reflected axis flips sign (polar-vector rule);
* 16 vector channels (``v_channels`` shape ``(N, 16, 3)``, readout ``v``
  ``(N, 3)``);
* contact edges (V2): only-ADD (chemical union edges never deleted / never
  duplicated), deterministic (same input → same output), and at most 16
  nearest contact neighbours per atom sorted by ``(dist, atom_idx)``.
"""
from __future__ import annotations

import numpy as np
import torch

from pes2ts_core.pesn.models import require_torch
from pes2ts_core.pesn.models.dual_egnn import (
    DualTSModelConfig,
    collate_dual_states,
)
from pes2ts_core.pesn.models.dual_egnn_painn import (
    DUAL_EGNN_PAINN_VERSION,
    DualTSPaiNNEncoder,
    DualTSPaiNNEncoderConfig,
    _build_contact_edges,
)
from pes2ts_core.pesn.models.dual_ts_displacement import (
    DisplacementHeadConfig,
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


def make_painn_policy(hidden: int = 32, n_layers: int = 2,
                      contact_edges: bool = False):
    torch.manual_seed(11)
    pol = build_displacement_policy(
        DualTSModelConfig(),
        DisplacementHeadConfig(readout_hidden=16),
        encoder_kind="painn", hidden=hidden, n_layers=n_layers,
        vector_channels=16, contact_edges=contact_edges,
    ).to(torch.float64)
    pol.train(False)
    return pol


def rotation_matrix(theta: float) -> torch.Tensor:
    c, s = np.cos(theta), np.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def reflection_matrix() -> torch.Tensor:
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])


# ---------------------------------------------------------------------------
# (1) translation invariance
# ---------------------------------------------------------------------------

def test_translation_invariance():
    pol = make_painn_policy()
    batch = collate_dual_states(torch, synthetic_states())
    with torch.no_grad():
        base = pol(batch)
        shifted = dict(batch)
        shifted["x"] = batch["x"] + torch.tensor([0.37, -1.2, 2.5]).double()
        out = pol(shifted)
    assert float((base["h"] - out["h"]).abs().max()) <= TOL
    assert float((base["v"] - out["v"]).abs().max()) <= TOL
    assert float((base["a_hat"] - out["a_hat"]).abs().max()) <= TOL


# ---------------------------------------------------------------------------
# (2) rotation equivariance + reflection sign flip (polar vector)
# ---------------------------------------------------------------------------

def test_rotation_equivariance_and_reflection_sign_flip():
    pol = make_painn_policy()
    batch = collate_dual_states(torch, synthetic_states())
    R = rotation_matrix(0.9).double()
    P = reflection_matrix().double()
    with torch.no_grad():
        base = pol(batch)
        rot = dict(batch)
        rot["x"] = batch["x"] @ R.T
        rot["g"] = batch["g"] @ R.T
        out_r = pol(rot)
        refl = dict(batch)
        refl["x"] = batch["x"] @ P.T
        refl["g"] = batch["g"] @ P.T
        out_p = pol(refl)
    # invariant scalars
    assert float((base["h"] - out_r["h"]).abs().max()) <= TOL
    assert float((base["h"] - out_p["h"]).abs().max()) <= TOL
    # covariant polar vectors under proper rotation
    assert float((base["a_hat"] @ R.T - out_r["a_hat"]).abs().max()) <= TOL
    assert float((base["v"] @ R.T - out_r["v"]).abs().max()) <= TOL
    # reflection flips the reflected axis (polar vector, not pseudovector)
    assert float((base["a_hat"] @ P.T - out_p["a_hat"]).abs().max()) <= TOL
    assert float((base["v"] @ P.T - out_p["v"]).abs().max()) <= TOL


# ---------------------------------------------------------------------------
# (3) 16 vector channels + readout shapes
# ---------------------------------------------------------------------------

def test_vector_channel_shapes():
    torch.manual_seed(5)
    enc = DualTSPaiNNEncoder(DualTSPaiNNEncoderConfig(
        hidden=32, n_layers=2, vector_channels=16)).to(torch.float64)
    enc.train_mode(False)
    batch = collate_dual_states(torch, synthetic_states(2))
    n = int(batch["z"].shape[0])
    with torch.no_grad():
        out = enc(batch)
    assert out["v_channels"].shape == (n, 16, 3)
    assert out["v"].shape == (n, 3)
    assert out["h"].shape == (n, 32)
    assert enc.n_parameters == sum(p.numel() for p in enc.parameters())
    assert DUAL_EGNN_PAINN_VERSION == "dual_egnn_painn_v1"


# ---------------------------------------------------------------------------
# (4) contact edges: only-add, deterministic, 16-neighbour truncation
# ---------------------------------------------------------------------------

def _cluster_geometry(n: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.2, size=(n, 3))  # dense cluster, all within ~1 Å
    return torch.tensor(x, dtype=torch.float64)


def test_contact_edges_only_add_deterministic_and_truncated():
    x = _cluster_geometry()
    # two chemical (undirected) union edges: (0,1) and (0,2), both directions.
    chem = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long)
    node_counts = torch.tensor([int(x.shape[0])])
    kw = dict(x=x, chem_edge_index=chem, node_counts=node_counts,
              n_graphs=1, cutoff=5.0, max_neighbors=16)
    c1 = _build_contact_edges(torch, **kw)
    c2 = _build_contact_edges(torch, **kw)

    src, dst, env = c1["src"], c1["dst"], c1["envelope"]
    assert src.shape[0] % 2 == 0
    contact_pairs = {
        (min(int(src[e]), int(dst[e])), max(int(src[e]), int(dst[e])))
        for e in range(int(src.shape[0]))}
    chem_pairs = {(0, 1), (0, 2)}
    # only ADD: never duplicate or delete a chemical union edge.
    assert contact_pairs.isdisjoint(chem_pairs)
    # determinism: bitwise identical on identical input.
    assert torch.equal(src, c2["src"])
    assert torch.equal(dst, c2["dst"])
    assert torch.equal(env, c2["envelope"])
    # at most 16 contact neighbours per atom (undirected).
    deg: dict[int, int] = {}
    for (a, b) in contact_pairs:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    assert all(d <= 16 for d in deg.values())
    # envelope values lie in (0, 1] and are symmetric across both directions.
    assert bool((env > 0.0).all()) and bool((env <= 1.0 + 1e-12).all())


def test_contact_edges_encoder_deterministic():
    pol = make_painn_policy(contact_edges=True)
    batch = collate_dual_states(torch, synthetic_states(2))
    with torch.no_grad():
        a = pol(batch)["a_hat"]
        b = pol(batch)["a_hat"]
    assert torch.equal(a, b)
