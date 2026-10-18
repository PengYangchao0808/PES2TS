"""Symmetry gates for the V5 sprint M2 Star EGNN (ADR-0026 §2.7/§5, G5.1).

Fixtures are hand-built ``KeyAtomBatch`` tensors (small molecules,
8–15 nodes) — no Lane A/B imports.  Gates (plan §3 M2 / ADR-0026 §5
verification obligation 1):

* param budget: ``sum(p.numel()) ≤ 90_000``;
* proper rotation equivariance (20 draws) + translation invariance:
  u(XQ+t) ≈ u(X)Q, relative error < 1e-5; z invariant < 1e-5;
* permutation equivariance: u rows follow the key permutation, z
  invariant;
* reflection (polar-vector contract): u(XQ) = u(X)Q for
  Q = diag(−1,1,1), z invariant;
* anchor-off batch: forward runs, same shapes, finite outputs.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pes2ts_core.pesn.contracts.key_atom import (
    EDGE_DIM,
    EDGE_SLOTS,
    NODE_INV_DIM,
    TASK_DIM,
    KeyAtomBatch,
)
from pes2ts_core.pesn.models import require_torch
from pes2ts_core.pesn.models.key_atom_egnn import (
    MEASURED_PARAMS,
    PARAM_BUDGET,
    STAR_EGNN_VERSION,
    StarEGNN,
)

torch = require_torch()

TOL = 1e-5
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Synthetic batch builder (hand-built tensors only)
# ---------------------------------------------------------------------------

def _graph_rows(n: int, rng: np.random.Generator):
    """One synthetic molecule: covalent chain + spatial edges < 4.5 Å."""
    pos = rng.uniform(-2.5, 2.5, size=(n, 3))
    cov = [(i, i + 1) for i in range(n - 1)]
    edges = []
    for a, b in cov:
        edges.append((a, b, "cov"))
        edges.append((b, a, "cov"))
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in cov or (j, i) in cov:
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < 4.5:
                edges.append((i, j, "rad"))
                edges.append((j, i, "rad"))
    return pos, edges


def make_batch(*, seed: int = 0, n_graphs: int = 2,
               anchor_off: bool = False) -> KeyAtomBatch:
    rng = np.random.default_rng(seed)
    zs, h_invs, poss, pos0s, anchors = [], [], [], [], []
    e_src, e_dst, e_attrs, r_ijs, r0_ijs = [], [], [], [], []
    task_rows, batch_vec, ptr = [], [], [0]
    key_index, key_ptr, pair_i, pair_j, pair_edit, degk = [], [0], [], [], [], []
    offset = 0
    key_offset = 0
    for g in range(n_graphs):
        n = int(rng.integers(8, 13))
        key_rows = sorted(rng.choice(n, size=3, replace=False).tolist())
        pairs = [(key_rows[0], key_rows[1]), (key_rows[1], key_rows[2])]
        pos, edges = _graph_rows(n, rng)
        pos0 = pos + rng.normal(scale=0.3, size=pos.shape)
        anchor = (pos - pos.mean(0)) - (pos0 - pos0.mean(0))

        zs.append(rng.integers(1, 10, size=n).astype(np.int64))
        h_invs.append(rng.normal(size=(n, NODE_INV_DIM)))
        poss.append(pos)
        pos0s.append(pos0)
        anchors.append(anchor)
        for a, b, kind in edges:
            e_src.append(a + offset)
            e_dst.append(b + offset)
            attr = np.zeros(EDGE_DIM)
            attr[EDGE_SLOTS["covalent_start"]] = 1.0 if kind == "cov" else 0.0
            attr[EDGE_SLOTS["covalent_other"]] = 1.0 if kind == "cov" else 0.0
            attr[EDGE_SLOTS["is_k"]] = 0.0
            e_attrs.append(attr)
            r_ijs.append(pos[a] - pos[b])
            r0_ijs.append(pos0[a] - pos0[b])
        # force key-pair edges into the edge list as covalent/K
        for a, b in pairs:
            for (x, y) in ((a, b), (b, a)):
                e_src.append(x + offset)
                e_dst.append(y + offset)
                attr = np.zeros(EDGE_DIM)
                attr[EDGE_SLOTS["is_k"]] = 1.0
                attr[EDGE_SLOTS["covalent_start"]] = 1.0
                attr[EDGE_SLOTS["covalent_other"]] = 1.0
                e_attrs.append(attr)
                r_ijs.append(pos[x] - pos[y])
                r0_ijs.append(pos0[x] - pos0[y])

        task = np.zeros(TASK_DIM)
        task[0 if g % 2 == 0 else 1] = 1.0
        task[16] = 0.08 / 0.08  # d0 slot
        task_rows.append(task)
        batch_vec.extend([g] * n)
        ptr.append(ptr[-1] + n)

        key_index.extend(k + offset for k in key_rows)
        key_ptr.append(key_ptr[-1] + len(key_rows))
        # GLOBAL node-row pair space (ADR-0026 §2.6)
        pair_i.extend(key_rows[i] + offset for i in range(len(key_rows) - 1))
        pair_j.extend(key_rows[i] + offset for i in range(1, len(key_rows)))
        pair_edit.extend([1.0 if i % 2 == 0 else -1.0
                          for i in range(len(key_rows) - 1)])
        degk.extend([2.0, 2.0, 1.0][:len(key_rows)])
        offset += n
        key_offset += len(key_rows)

    batch = KeyAtomBatch(
        z_atomic=torch.as_tensor(np.concatenate(zs), dtype=torch.int64),
        h_node_inv=torch.as_tensor(np.concatenate(h_invs), dtype=DTYPE),
        pos=torch.as_tensor(np.concatenate(poss), dtype=DTYPE),
        pos0=torch.as_tensor(np.concatenate(pos0s), dtype=DTYPE),
        anchor_vec=torch.as_tensor(np.concatenate(anchors), dtype=DTYPE),
        edge_index=torch.as_tensor(np.stack([e_src, e_dst]), dtype=torch.int64),
        edge_attr=torch.as_tensor(np.stack(e_attrs), dtype=DTYPE),
        r_ij=torch.as_tensor(np.stack(r_ijs), dtype=DTYPE),
        r0_ij=torch.as_tensor(np.stack(r0_ijs), dtype=DTYPE),
        task19=torch.as_tensor(np.stack(task_rows), dtype=DTYPE),
        batch=torch.as_tensor(batch_vec, dtype=torch.int64),
        ptr=torch.as_tensor(ptr, dtype=torch.int64),
        key_index=torch.as_tensor(key_index, dtype=torch.int64),
        key_ptr=torch.as_tensor(key_ptr, dtype=torch.int64),
        pair_index=torch.as_tensor(np.stack([pair_i, pair_j]), dtype=torch.int64),
        pair_edit=torch.as_tensor(pair_edit, dtype=DTYPE),
        degK=torch.as_tensor(degk, dtype=DTYPE),
    )
    if anchor_off:
        batch.anchor_off_()
    return batch


def make_model(seed: int = 11) -> StarEGNN:
    torch.manual_seed(seed)
    model = StarEGNN()
    model.to(DTYPE)
    model.eval()
    return model


def fwd(model: StarEGNN, batch: KeyAtomBatch):
    """Eval-mode forward without building a graph (symmetry gates)."""
    with torch.no_grad():
        return model(batch)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def apply_rotation(batch: KeyAtomBatch, Q: np.ndarray,
                   t: np.ndarray | None = None) -> KeyAtomBatch:
    """Rotate/translate every polar vector of a FRESH batch consistently."""
    Qt = torch.as_tensor(Q, dtype=DTYPE)
    tt = (torch.as_tensor(t, dtype=DTYPE) if t is not None
          else torch.zeros(3, dtype=DTYPE))
    batch.pos = batch.pos @ Qt + tt
    batch.pos0 = batch.pos0 @ Qt + tt
    batch.anchor_vec = batch.anchor_vec @ Qt
    batch.r_ij = batch.r_ij @ Qt
    batch.r0_ij = batch.r0_ij @ Qt
    return batch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_param_budget_and_version(capsys):
    model = make_model()
    n = sum(p.numel() for p in model.parameters())
    with capsys.disabled():
        print(f"\n[{STAR_EGNN_VERSION}] measured params = {n}")
    assert n <= PARAM_BUDGET
    assert n == MEASURED_PARAMS, (
        f"MEASURED_PARAMS constant {MEASURED_PARAMS} stale (actual {n})")


def test_proper_rotation_equivariance_and_translation():
    model = make_model()
    rng = np.random.default_rng(20261018)
    base = make_batch(seed=3)
    u0, z0 = fwd(model, base)
    m = u0.shape[0]
    denom = max(float(u0.norm()), 0.08 * math.sqrt(m))
    for _ in range(20):
        Q = random_rotation(rng)
        t = rng.uniform(-10, 10, size=3)
        batch = apply_rotation(make_batch(seed=3), Q, t)
        u1, z1 = fwd(model, batch)
        u_rot = u0 @ torch.as_tensor(Q, dtype=DTYPE)
        err = float((u1 - u_rot).norm()) / denom
        assert err < TOL, f"rotation relative error {err}"
        z_err = float((z1 - z0).norm()) / max(float(z0.norm()), 1e-8)
        assert z_err < TOL, f"z invariance error {z_err}"


def test_permutation_equivariance():
    model = make_model()
    rng = np.random.default_rng(7)
    base = make_batch(seed=5)
    u0, z0 = fwd(model, base)

    # gather convention: new_row[i] = old_row[perm[i]]; an old row r now
    # lives at new row inv[r]
    perm_local = []
    offsets = base.ptr.numpy()
    for g in range(len(offsets) - 1):
        n = offsets[g + 1] - offsets[g]
        perm_local.append(rng.permutation(n) + offsets[g])
    perm = np.concatenate(perm_local)
    p = torch.as_tensor(perm)
    inv = torch.empty_like(p)
    inv[p] = torch.arange(len(perm))

    b = make_batch(seed=5)
    b.z_atomic = b.z_atomic[p]
    b.h_node_inv = b.h_node_inv[p]
    b.pos = b.pos[p]
    b.pos0 = b.pos0[p]
    b.anchor_vec = b.anchor_vec[p]
    b.batch = b.batch[p]
    b.edge_index = torch.stack([inv[b.edge_index[0]], inv[b.edge_index[1]]])
    b.key_index = inv[b.key_index]

    u1, z1 = fwd(model, b)
    # key slots keep their order; old key row r lives at new row inv[r]
    assert torch.allclose(u1, u0, atol=TOL)
    z_err = float((z1 - z0).norm()) / max(float(z0.norm()), 1e-8)
    assert z_err < TOL


def test_permutation_equivariance_key_slot_shuffle():
    """Shuffle the key_index slot ORDER too: u rows must follow."""
    model = make_model()
    rng = np.random.default_rng(8)
    base = make_batch(seed=6)
    u0, _ = fwd(model, base)
    b = make_batch(seed=6)
    n_keys = int(b.key_index.numel())
    n_nodes = int(b.pos.shape[0])
    node_to_slot = torch.full((n_nodes,), -1, dtype=torch.int64)
    node_to_slot[b.key_index] = torch.arange(n_keys)
    old_slots = node_to_slot[b.pair_index]
    slot_perm = torch.as_tensor(rng.permutation(n_keys))
    b.key_index = b.key_index[slot_perm]
    b.degK = b.degK[slot_perm]
    # pair_index is in GLOBAL node-row space: translate through the old
    # slot map, apply slot_perm^{-1}, then back to the new global rows.
    inv = torch.empty(n_keys, dtype=torch.int64)
    inv[slot_perm] = torch.arange(n_keys)
    b.pair_index = b.key_index[inv[old_slots]]
    u1, _ = fwd(model, b)
    assert torch.allclose(u1, u0[slot_perm], atol=TOL)


def test_reflection_polar_contract():
    model = make_model()
    Q = np.diag([-1.0, 1.0, 1.0])
    base = make_batch(seed=9)
    u0, z0 = fwd(model, base)
    batch = apply_rotation(make_batch(seed=9), Q)
    u1, z1 = fwd(model, batch)
    u_expected = u0 @ torch.as_tensor(Q, dtype=DTYPE)
    err = float((u1 - u_expected).norm()) / max(
        float(u0.norm()), 0.08 * math.sqrt(u0.shape[0]))
    assert err < TOL, f"reflection equivariance error {err}"
    z_err = float((z1 - z0).norm()) / max(float(z0.norm()), 1e-8)
    assert z_err < TOL


def test_anchor_off_batch():
    model = make_model()
    on = make_batch(seed=13, anchor_off=False)
    off = make_batch(seed=13, anchor_off=True)
    u_on, z_on = fwd(model, on)
    u_off, z_off = fwd(model, off)
    assert u_off.shape == u_on.shape
    assert z_off.shape == z_on.shape
    assert bool(torch.isfinite(u_off).all())
    assert bool(torch.isfinite(z_off).all())
    # anchor-off must change the output (the anchor channel was live)
    assert float((u_off - u_on).norm()) > 1e-8


def test_forward_shapes_and_grad():
    model = make_model()
    batch = make_batch(seed=17)
    u, z = model(batch)
    n_keys = int(batch.key_index.numel())
    n_graphs = int(batch.ptr.numel()) - 1
    assert u.shape == (n_keys, 3)
    assert z.shape == (n_graphs, 64)
    loss = (u ** 2).sum() + (z ** 2).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)


def test_z_padding_rejected():
    model = make_model()
    batch = make_batch(seed=19)
    batch.z_atomic[0] = 0
    with pytest.raises(Exception) as excinfo:
        model(batch)
    assert "Z_PADDING" in str(excinfo.value)
    batch.z_atomic[0] = 1
    batch.z_atomic[1] = 200
    with pytest.raises(Exception) as excinfo:
        model(batch)
    assert "Z_OUT_OF_RANGE" in str(excinfo.value)
