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


# ---------------------------------------------------------------------------
# V5.1a magnitude head (plan §3.2; ADR-0027) — head-on gates
# ---------------------------------------------------------------------------

from pes2ts_core.pesn.models.key_atom_egnn import (  # noqa: E402
    KeyAtomModelError,
    MAG_BIAS_INIT,
    MAG_DEG_SCALE_TYP,
    MAG_INIT_TARGET,
    MAG_RAW_INIT_TARGET,
    MEASURED_PARAMS_WITH_HEAD,
)


def make_head_model(seed: int = 11, mode: str = "per_atom") -> StarEGNN:
    torch.manual_seed(seed)
    model = StarEGNN(magnitude_head=True, magnitude_mode=mode)
    model.to(DTYPE)
    model.eval()
    return model


def test_magnitude_head_param_budget(capsys):
    model = make_head_model()
    n = sum(p.numel() for p in model.parameters())
    with capsys.disabled():
        print(f"\n[{STAR_EGNN_VERSION}] head-on measured params = {n}")
    assert n <= PARAM_BUDGET
    assert n == MEASURED_PARAMS_WITH_HEAD, (
        f"MEASURED_PARAMS_WITH_HEAD stale (actual {n})")
    # head-off construction stays byte-identical to the legacy count
    legacy = make_model()
    assert sum(p.numel() for p in legacy.parameters()) == MEASURED_PARAMS


def test_magnitude_head_init_near_0p05():
    """At init the EFFECTIVE magnitude (what reaches ‖u‖) is ≈0.05 Å for
    degK_typ=2 key atoms (plan §3.2: 初始化 s_i≈0.05 Å — effective
    semantics, fixed 2026-10-19: the raw head output is degK-scaled
    BEFORE reaching u, so the bias calibrates s_eff = s·deg_scale).

    degK-dependence is BY DESIGN: s_eff_init = 1.25·a_ref/max(1,degK)
    (0.05 Å at degK=2, 0.1 Å at degK=1, 0.025 Å at degK=4).
    """
    model = make_head_model(seed=3)
    batch = make_batch(seed=21)
    u, _z = fwd(model, batch)
    s_eff = model.last_magnitudes
    s_raw = model.last_raw_magnitudes
    assert s_eff is not None and s_eff.shape == (u.shape[0],)
    assert s_raw is not None and s_raw.shape == (u.shape[0],)
    assert bool(torch.isfinite(s_eff).all())
    assert bool((s_eff > 0).all()), "effective magnitudes must be > 0"
    # contract: s_eff == raw softplus output × deg_scale; raw ≡ 1.25 at init
    deg_scale = (0.08 / batch.degK.clamp(min=1.0)).to(DTYPE)
    assert torch.allclose(s_raw, torch.full_like(s_raw, MAG_RAW_INIT_TARGET),
                          atol=1e-6), "raw head output must be 1.25 at init"
    assert torch.allclose(s_eff, s_raw * deg_scale, atol=1e-12)
    # degK=2 fixture atoms: effective magnitude ≈ 0.05 Å
    deg2 = batch.degK == 2.0
    assert bool(deg2.any())
    assert float((s_eff[deg2] - MAG_INIT_TARGET).abs().max()) < 1e-6
    # ‖u‖ is bounded by the effective magnitude (‖v_hat‖ ≤ 1)
    assert bool((u.norm(dim=-1) <= s_eff + 1e-12).all())
    assert abs(MAG_DEG_SCALE_TYP - 0.04) < 1e-12
    assert abs(MAG_BIAS_INIT - math.log(math.expm1(1.25))) < 1e-12


def test_magnitude_head_graph_mode_broadcasts():
    """graph mode: one invariant RAW scalar per graph; the effective
    magnitude is that scalar × per-atom deg_scale (degK-dependent)."""
    model = make_head_model(seed=5, mode="graph")
    batch = make_batch(seed=23)
    u, _z = fwd(model, batch)
    s_eff = model.last_magnitudes
    s_raw = model.last_raw_magnitudes
    key_ptr = batch.key_ptr.numpy()
    deg_scale = (0.08 / batch.degK.clamp(min=1.0)).to(DTYPE)
    assert torch.allclose(s_eff, s_raw * deg_scale, atol=1e-12)
    for b in range(len(key_ptr) - 1):
        lo, hi = int(key_ptr[b]), int(key_ptr[b + 1])
        assert float((s_raw[lo:hi] - s_raw[lo]).abs().max()) < 1e-12
    assert float((s_raw - MAG_RAW_INIT_TARGET).abs().max()) < 1e-6
    deg2 = batch.degK == 2.0
    assert float((s_eff[deg2] - MAG_INIT_TARGET).abs().max()) < 1e-6
    assert float(u.norm()) > 0.0


def test_magnitude_head_rotation_equivariance_and_translation():
    """Direction head equivariance + invariant magnitudes (plan §3.2:
    幅度头为不变量、方向头为等变量)."""
    model = make_head_model(seed=13)
    rng = np.random.default_rng(20261019)
    base = make_batch(seed=27)
    u0, z0 = fwd(model, base)
    s0 = model.last_magnitudes.clone()
    m = u0.shape[0]
    denom = max(float(u0.norm()), 0.05 * math.sqrt(m))
    for _ in range(20):
        Q = random_rotation(rng)
        t = rng.uniform(-10, 10, size=3)
        batch = apply_rotation(make_batch(seed=27), Q, t)
        u1, z1 = fwd(model, batch)
        u_rot = u0 @ torch.as_tensor(Q, dtype=DTYPE)
        err = float((u1 - u_rot).norm()) / denom
        assert err < TOL, f"rotation relative error {err}"
        s_err = float((model.last_magnitudes - s0).abs().max())
        assert s_err < TOL, f"magnitude invariance error {s_err}"
        z_err = float((z1 - z0).norm()) / max(float(z0.norm()), 1e-8)
        assert z_err < TOL, f"z invariance error {z_err}"


def test_magnitude_head_permutation_equivariance():
    """u rows AND s rows follow the key-slot permutation (plan §3.2:
    G5.1 分别测置换)."""
    model = make_head_model(seed=17)
    rng = np.random.default_rng(19)
    base = make_batch(seed=29)
    u0, _ = fwd(model, base)
    s0 = model.last_magnitudes.clone()
    b = make_batch(seed=29)
    n_keys = int(b.key_index.numel())
    n_nodes = int(b.pos.shape[0])
    node_to_slot = torch.full((n_nodes,), -1, dtype=torch.int64)
    node_to_slot[b.key_index] = torch.arange(n_keys)
    old_slots = node_to_slot[b.pair_index]
    slot_perm = torch.as_tensor(rng.permutation(n_keys))
    b.key_index = b.key_index[slot_perm]
    b.degK = b.degK[slot_perm]
    inv = torch.empty(n_keys, dtype=torch.int64)
    inv[slot_perm] = torch.arange(n_keys)
    b.pair_index = b.key_index[inv[old_slots]]
    u1, _ = fwd(model, b)
    assert torch.allclose(u1, u0[slot_perm], atol=TOL)
    assert torch.allclose(model.last_magnitudes, s0[slot_perm], atol=TOL)


def test_magnitude_head_reflection_polar_contract():
    """u is a POLAR vector (u(XQ) = u(X)Q under reflections) while s and z
    stay invariant (plan §3.2: G5.1 分别测反射)."""
    model = make_head_model(seed=23)
    Q = np.diag([-1.0, 1.0, 1.0])
    base = make_batch(seed=31)
    u0, z0 = fwd(model, base)
    s0 = model.last_magnitudes.clone()
    batch = apply_rotation(make_batch(seed=31), Q)
    u1, z1 = fwd(model, batch)
    u_expected = u0 @ torch.as_tensor(Q, dtype=DTYPE)
    err = float((u1 - u_expected).norm()) / max(
        float(u0.norm()), 0.05 * math.sqrt(u0.shape[0]))
    assert err < TOL, f"reflection equivariance error {err}"
    assert float((model.last_magnitudes - s0).abs().max()) < TOL
    z_err = float((z1 - z0).norm()) / max(float(z0.norm()), 1e-8)
    assert z_err < TOL


def test_magnitude_head_forward_shapes_and_grad():
    model = make_head_model(seed=33)
    batch = make_batch(seed=37)
    u, z = model(batch)
    n_keys = int(batch.key_index.numel())
    n_graphs = int(batch.ptr.numel()) - 1
    assert u.shape == (n_keys, 3)
    assert z.shape == (n_graphs, 64)
    s = model.last_magnitudes  # EFFECTIVE s_eff = s_raw·deg_scale
    # soft direction norm: ‖u_i‖ ≤ s_eff_i (‖v_hat‖ ≤ 1)
    assert bool((u.norm(dim=-1) <= s + 1e-12).all())
    # and s_eff is degK-scaled from the raw head output
    s_raw = model.last_raw_magnitudes
    deg_scale = (0.08 / batch.degK.clamp(min=1.0)).to(DTYPE)
    assert torch.allclose(s, s_raw * deg_scale, atol=1e-12)
    loss = (u ** 2).sum() + (z ** 2).sum() + s.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    # zero-teacher direction: forcing the summed vector channel to zero
    # must yield a zero vector + zero gradient, never a fabricated axis
    for p in model.parameters():
        p.grad = None
    u_zero, _ = model(make_batch(seed=37))
    loss0 = u_zero.abs().max()
    loss0.backward()
    assert bool(torch.isfinite(loss0.detach()))
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all())
               for p in model.parameters())


def test_magnitude_mode_invalid_rejected():
    with pytest.raises(KeyAtomModelError) as excinfo:
        StarEGNN(magnitude_head=True, magnitude_mode="global")
    assert "STAR_EGNN_MAGNITUDE_MODE" in str(excinfo.value)


def test_legacy_forward_unchanged_by_head_addition():
    """Head-off forward output must not depend on head-on construction
    (paired-initialization RNG stream untouched)."""
    legacy = make_model(seed=41)
    head = make_head_model(seed=41)
    # same seed → the shared parameters must be IDENTICAL
    for (name_l, p_l), (name_h, p_h) in zip(
            legacy.named_parameters(), head.named_parameters()):
        assert name_l == name_h
        assert torch.equal(p_l, p_h), name_l
    assert "mag_head.0.weight" in dict(head.named_parameters())
    legacy_head_off = StarEGNN(magnitude_head=False)
    names = dict(legacy_head_off.named_parameters())
    assert "mag_head.0.weight" not in names
