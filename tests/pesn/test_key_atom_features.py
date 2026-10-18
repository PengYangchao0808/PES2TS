"""Unit tests for the V5 key-atom feature lane (ADR-0026 §2.6, plan §2/§5.2).

Lane A's ``KeyAtomTask`` loader / ``weighted_kabsch`` may not be importable
yet; these tests drive the builders through a small duck-typed FakeTask that
implements exactly the surface documented in
:mod:`pes2ts_core.pesn.features.key_atom_graph`.  Integration with the real
Lane A types is exercised in the campaign dry-run (documented deviation).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom import (  # noqa: E402
    EDGE_ANCHOR_SLOTS,
    EDGE_DIM,
    EDGE_SLOTS,
    NODE_ANCHOR_SLOTS,
    NODE_INV_DIM,
    NODE_INV_SLOTS,
    TASK_DIM,
    KeyAtomBatch,
)
from pes2ts_core.pesn.features.key_atom_graph import (  # noqa: E402
    build_sample_features,
    build_task19,
    collate,
)
from pes2ts_core.pesn.features.key_atom_spec import (  # noqa: E402
    HOP_SLOTS,
    METHOD_VOCAB_SIZE,
    N_RBF,
    RBF_CUTOFF,
    SOLVENT_VOCAB_SIZE,
    SPATIAL_MAX_NEIGHBORS,
    SPATIAL_TIE_TOL,
    build_vocab,
    gaussian_rbf,
    load_feature_spec,
    pair_r_eq,
    pauling_proxy,
    stable_id_hash,
    vocab_index,
    write_feature_spec,
)


# ---------------------------------------------------------------------------
# Synthetic task fixtures
# ---------------------------------------------------------------------------
class FakeBond:
    def __init__(self, i: int, j: int, order: float, aromatic: bool = False):
        self.i = int(i)
        self.j = int(j)
        self.bond_order = float(order)
        self.aromatic = bool(aromatic)


class FakeGraph:
    def __init__(self, bonds, formal_charges=None):
        self.bonds = tuple(bonds)
        self.formal_charges = formal_charges


def make_task(atomic_numbers, key_rows, bonds_start, bonds_other,
              charge=0, multiplicity=1, **kw):
    """Duck-typed KeyAtomTask stand-in (see module docstring)."""
    return SimpleNamespace(
        atomic_numbers=np.asarray(atomic_numbers, dtype=np.int64),
        key_rows=tuple(key_rows),
        endpoints=(FakeGraph(bonds_start), FakeGraph(bonds_other)),
        charge=charge,
        multiplicity=multiplicity,
        reaction_id=kw.get("reaction_id", "rxn_fake"),
        reaction_group_id=kw.get("group_id", "grp_fake"),
        method_id=kw.get("method_id", "B3LYP"),
        solvent_id=kw.get("solvent_id", None),
    )


def h_transfer_task():
    """H2O-like toy: H0 migrates O1 -> O2 (rows: 0=H, 1=O, 2=O, 3=H cap)."""
    z = (1, 8, 8, 1)
    start = [FakeBond(0, 1, 1.0), FakeBond(1, 2, 1.0), FakeBond(2, 3, 1.0)]
    other = [FakeBond(0, 2, 1.0), FakeBond(1, 2, 1.0), FakeBond(2, 3, 1.0)]
    return make_task(z, key_rows=(0, 2), bonds_start=start, bonds_other=other)


def geometry(task, seed=0):
    n = len(task.atomic_numbers)
    rng = np.random.default_rng(seed)
    pos = rng.normal(scale=0.6, size=(n, 3))
    # keep the toy chemically sane: pull bonded partners closer
    for view in task.endpoints:
        for b in view.bonds:
            mid = (pos[b.i] + pos[b.j]) / 2
            pos[b.i] = mid + (pos[b.i] - mid) * 0.35
            pos[b.j] = mid + (pos[b.j] - mid) * 0.35
    pos0 = pos + rng.normal(scale=0.05, size=(n, 3))
    return pos.astype(np.float64), pos0.astype(np.float64)


def build(task=None, *, pos=None, pos0=None, branch=0, d0=0.08, anchor=True,
          seed=0, **kw):
    task = task if task is not None else h_transfer_task()
    if pos is None or pos0 is None:
        pos, pos0 = geometry(task, seed)
    method_hash = kw.pop("method_hash", stable_id_hash("B3LYP"))
    solvent_hash = kw.pop("solvent_hash", 0)
    return build_sample_features(
        task, pos, pos0, branch, d0, anchor, method_hash, solvent_hash,
        rho_known=kw.pop("rho_known", True),
        solvent_known=kw.pop("solvent_known", False), **kw)


EMPTY_VOCAB: dict[int, int] = {}


# ---------------------------------------------------------------------------
# Dimensions & basic sanity
# ---------------------------------------------------------------------------
def test_dims_and_rbf_sanity():
    s = build()
    n = s["h_node_inv"].shape[0]
    assert s["h_node_inv"].shape == (n, NODE_INV_DIM)
    assert s["h_node_inv"].dtype == np.float32
    e = s["edge_attr"].shape[0]
    assert s["edge_attr"].shape == (e, EDGE_DIM)
    assert s["edge_index"].shape == (2, e)
    assert s["r_ij"].shape == (e, 3)

    # RBF block: 16 channels, values in (0, 1], sane sums for r in range
    rbf = s["edge_attr"][:, EDGE_SLOTS["rbf0"]:EDGE_SLOTS["rbf0"] + N_RBF]
    assert rbf.shape[1] == 16
    assert np.all(rbf >= 0.0) and np.all(rbf <= 1.0 + 1e-6)
    sums = rbf.sum(axis=1)
    assert np.all(sums > 0.1)
    # a bonded pair at ~1 Å must hit the near-centre expansion hard
    on = gaussian_rbf(np.array([1.0]))
    assert on.max() > 0.9 and on.sum() > 1.0
    assert RBF_CUTOFF == 4.5

    # task19
    t = build_task19(s, EMPTY_VOCAB, EMPTY_VOCAB)
    assert t.shape == (TASK_DIM,)
    # hop one-hot exactly one-hot over the 5 bins
    hop = s["h_node_inv"][:, list(HOP_SLOTS)]
    assert np.all(hop.sum(axis=1) == pytest.approx(1.0))
    assert np.all((hop == 0.0) | (hop == 1.0))

    # pair/key tables: K = changed pairs over key atoms
    m = s["key_index"].shape[0]
    assert s["degK"].shape == (m,)
    assert s["pair_index"].shape[1] >= 0
    # every K pair has both endpoints in key_index and a nonzero |dB|
    key_set = set(int(a) for a in s["key_index"])
    for p in range(s["pair_index"].shape[1]):
        i, j = int(s["pair_index"][0, p]), int(s["pair_index"][1, p])
        assert i in key_set and j in key_set and i < j
        assert float(s["pair_edit"][p]) != 0.0


def test_pauling_proxy_table():
    assert pair_r_eq(1, 1) == pytest.approx(0.64)     # H–H
    assert pair_r_eq(6, 8) == pytest.approx(0.69 + 0.63)
    assert pair_r_eq(26, 26) is None                   # Fe absent -> mask 0
    val, mask = pauling_proxy(1.4, 6, 6)
    assert mask == 1.0 and val > 0.0
    val, mask = pauling_proxy(1.4, 26, 6)
    assert (val, mask) == (0.0, 0.0)


def test_spec_writer_loader(tmp_path):
    path = tmp_path / "spec.json"
    write_feature_spec(path, {"task:rxn": "ab" * 32})
    doc = load_feature_spec(path)
    assert doc["schema_version"] == "pesn_v5_feature_spec_v1"
    assert doc["node_inv_dim"] == NODE_INV_DIM
    assert doc["edge_dim"] == EDGE_DIM


# ---------------------------------------------------------------------------
# Anchor masking (plan §5.2)
# ---------------------------------------------------------------------------
def test_anchor_masking_collate():
    samples = [build(seed=1), build(seed=2)]
    on = collate(samples, "cpu", True, EMPTY_VOCAB, EMPTY_VOCAB)
    off = collate(samples, "cpu", False, EMPTY_VOCAB, EMPTY_VOCAB)

    # shapes preserved exactly
    assert on.h_node_inv.shape == off.h_node_inv.shape
    assert on.edge_attr.shape == off.edge_attr.shape
    assert on.r0_ij.shape == off.r0_ij.shape
    assert on.pos0.shape == on.anchor_vec.shape == on.pos.shape

    # anchor slots zeroed
    assert torch.all(off.h_node_inv[:, list(NODE_ANCHOR_SLOTS)] == 0.0)
    assert torch.all(off.edge_attr[:, list(EDGE_ANCHOR_SLOTS)] == 0.0)
    assert torch.all(off.r0_ij == 0.0)
    assert torch.all(off.anchor_vec == 0.0)
    assert torch.all(off.pos0 == 0.0)

    # non-anchor slots bit-identical to the anchor-on batch
    node_mask = [i for i in range(NODE_INV_DIM) if i not in NODE_ANCHOR_SLOTS]
    edge_mask = [i for i in range(EDGE_DIM) if i not in EDGE_ANCHOR_SLOTS]
    assert torch.equal(on.h_node_inv[:, node_mask], off.h_node_inv[:, node_mask])
    assert torch.equal(on.edge_attr[:, edge_mask], off.edge_attr[:, edge_mask])
    assert torch.equal(on.pos, off.pos)
    assert torch.equal(on.r_ij, off.r_ij)
    assert torch.equal(on.task19, off.task19)
    assert torch.equal(on.edge_index, off.edge_index)


def test_anchor_off_at_builder_level():
    on = build(anchor=True)
    off = build(anchor=False)
    assert np.all(off["h_node_inv"][:, list(NODE_ANCHOR_SLOTS)] == 0.0)
    assert np.all(off["edge_attr"][:, list(EDGE_ANCHOR_SLOTS)] == 0.0)
    assert np.all(off["pos0"] == 0.0)
    keep_n = [i for i in range(NODE_INV_DIM) if i not in NODE_ANCHOR_SLOTS]
    assert np.array_equal(on["h_node_inv"][:, keep_n], off["h_node_inv"][:, keep_n])


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------
def test_edges_bidirectional_symmetric():
    s = build()
    ei, ea = s["edge_index"], s["edge_attr"]
    e = ei.shape[1]
    assert e % 2 == 0
    src, dst = ei[0], ei[1]
    # pairs come in (i->j, j->i) couples with identical attrs
    for p in range(0, e, 2):
        assert (src[p], dst[p]) == (dst[p + 1], src[p + 1])
        assert np.array_equal(ea[p], ea[p + 1])
        assert np.allclose(s["r_ij"][p], -s["r_ij"][p + 1])
        assert np.allclose(s["r0_ij"][p], -s["r0_ij"][p + 1])


def test_forced_covalent_edge_beyond_cutoff():
    # stretch the O1-O2 bond way past 4.5 Å: the covalent edge MUST survive
    task = h_transfer_task()
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [7.0, 0.0, 0.0],
                    [7.9, 0.0, 0.0]])
    s = build(task, pos=pos, pos0=pos.copy())
    ei, ea = s["edge_index"], s["edge_attr"]
    found = False
    for p in range(ei.shape[1]):
        i, j = int(ei[0, p]), int(ei[1, p])
        if {i, j} == {1, 2}:
            found = True
            assert ea[p, EDGE_SLOTS["covalent_start"]] == 1.0
            assert ea[p, EDGE_SLOTS["is_radius"]] == 0.0
    assert found, "stretched covalent edge (1,2) missing"


def test_forced_k_edge_beyond_cutoff():
    # key atoms 0 and 2 bonded in START only (broken, dB=-1), pushed to 9 Å:
    # the K edge MUST survive the 4.5 Å cutoff
    z = (1, 8, 8)
    start = [FakeBond(0, 2, 1.0), FakeBond(0, 1, 1.0)]
    other = [FakeBond(1, 2, 1.0)]
    task = make_task(z, key_rows=(0, 2), bonds_start=start, bonds_other=other)
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    s = build(task, pos=pos, pos0=pos.copy())
    ei, ea = s["edge_index"], s["edge_attr"]
    found = False
    for p in range(ei.shape[1]):
        i, j = int(ei[0, p]), int(ei[1, p])
        if {i, j} == {0, 2}:
            found = True
            assert ea[p, EDGE_SLOTS["is_k"]] == 1.0
            assert ea[p, EDGE_SLOTS["covalent_start"]] == 1.0
            assert ea[p, EDGE_SLOTS["is_radius"]] == 0.0
    assert found, "forced K edge (0,2) missing"
    # K = changed pairs: (0,2) is the only changed pair with BOTH endpoints
    # key ((0,1) changed too but atom 1 is not key) => one pair, dB = -1
    assert s["pair_index"].shape[1] == 1
    assert int(s["pair_index"][0][0]) == 0 and int(s["pair_index"][1][0]) == 2
    assert float(s["pair_edit"][0]) == pytest.approx(-1.0)
    # degK: atom 0 and atom 2 each sit on exactly one changed pair
    assert float(s["degK"][0]) == 1.0 and float(s["degK"][1]) == 1.0


def test_spatial_cap_and_ties():
    # centre atom 0 with 40 equidistant neighbours at ~2.0 Å (no bonds)
    n_extra = 40
    pos = np.zeros((1 + n_extra, 3))
    for k in range(n_extra):
        ang = 2 * np.pi * k / n_extra
        pos[1 + k] = [2.0 * np.cos(ang), 2.0 * np.sin(ang), 0.0]
    # jitter distances well below the tie tolerance so they all tie
    rng = np.random.default_rng(7)
    pos[1:, 2] = rng.uniform(-5e-7, 5e-7, size=n_extra)
    z = (6,) + (1,) * n_extra
    task = make_task(z, key_rows=(0,), bonds_start=[], bonds_other=[])
    s = build(task, pos=pos, pos0=pos.copy())
    ei, ea = s["edge_index"], s["edge_attr"]
    centre_edges = [p for p in range(ei.shape[1])
                    if int(ei[0, p]) == 0 and ea[p, EDGE_SLOTS["is_radius"]] == 1.0]
    # every tied neighbour kept from the centre despite the 32 cap (the
    # reverse direction of each pair is a separate edge row)
    assert len(centre_edges) == n_extra
    assert len({int(ei[1, p]) for p in centre_edges}) == n_extra


def test_spatial_cap_strict():
    # 40 neighbours at DISTINCT distances: centre keeps only 32
    n_extra = 40
    pos = np.zeros((1 + n_extra, 3))
    for k in range(n_extra):
        pos[1 + k] = [1.5 + 0.05 * k, 0.0, float(k)]
    z = (6,) + (1,) * n_extra
    task = make_task(z, key_rows=(0,), bonds_start=[], bonds_other=[])
    s = build(task, pos=pos, pos0=pos.copy())
    ei, ea = s["edge_index"], s["edge_attr"]
    centre = [p for p in range(ei.shape[1])
              if int(ei[0, p]) == 0 and ea[p, EDGE_SLOTS["is_radius"]] == 1.0]
    assert len(centre) <= SPATIAL_MAX_NEIGHBORS * 2
    assert SPATIAL_TIE_TOL == 1e-6


# ---------------------------------------------------------------------------
# Permutation equivariance
# ---------------------------------------------------------------------------
def test_permutation_equivariance():
    task = h_transfer_task()
    pos, pos0 = geometry(task, seed=3)
    base = build(task, pos=pos, pos0=pos0)

    perm = np.array([2, 0, 3, 1])           # new_row = position of old row
    inv = np.argsort(perm)

    def remap_bonds(bonds):
        return [FakeBond(int(perm[b.i]), int(perm[b.j]), b.bond_order, b.aromatic)
                for b in bonds]

    task_p = make_task(
        tuple(int(task.atomic_numbers[inv[r]]) for r in range(4)),
        key_rows=tuple(sorted(int(perm[a]) for a in task.key_rows)),
        bonds_start=remap_bonds(task.endpoints[0].bonds),
        bonds_other=remap_bonds(task.endpoints[1].bonds),
    )
    pos_p = pos[inv]
    pos0_p = pos0[inv]
    permuted = build(task_p, pos=pos_p, pos0=pos0_p)

    # node rows permute
    assert np.array_equal(permuted["z_atomic"], task_p.atomic_numbers)
    assert np.allclose(permuted["h_node_inv"], base["h_node_inv"][inv],
                       atol=1e-6)
    # pos/pos0/anchor_vec permute (centroids are permutation-invariant)
    assert np.allclose(permuted["pos"], base["pos"][inv], atol=1e-6)
    assert np.allclose(permuted["anchor_vec"], base["anchor_vec"][inv], atol=1e-6)

    # edge SETS correspond under the permutation (order may differ)
    def edge_map(s):
        out = {}
        for p in range(s["edge_index"].shape[1]):
            i, j = int(s["edge_index"][0, p]), int(s["edge_index"][1, p])
            out[(i, j)] = (s["edge_attr"][p], s["r_ij"][p])
        return out

    e_base, e_perm = edge_map(base), edge_map(permuted)
    assert set(e_perm.keys()) == {(int(perm[i]), int(perm[j]))
                                  for (i, j) in e_base}
    for (i, j), (attr, rij) in e_base.items():
        pattr, prij = e_perm[(int(perm[i]), int(perm[j]))]
        assert np.allclose(pattr, attr, atol=1e-6)
        assert np.allclose(prij, rij, atol=1e-6)

    # pair_edit follows the pair permutation
    def pair_map(s):
        out = {}
        for p in range(s["pair_index"].shape[1]):
            i, j = int(s["pair_index"][0, p]), int(s["pair_index"][1, p])
            out[(i, j)] = float(s["pair_edit"][p])
        return out

    pb, pp = pair_map(base), pair_map(permuted)
    assert set(pp.keys()) == {(int(perm[i]), int(perm[j])) for (i, j) in pb}
    for (i, j), v in pb.items():
        assert pp[(int(perm[i]), int(perm[j]))] == pytest.approx(v)

    # task19 unchanged
    t1 = build_task19(base, EMPTY_VOCAB, EMPTY_VOCAB)
    t2 = build_task19(permuted, EMPTY_VOCAB, EMPTY_VOCAB)
    assert np.array_equal(t1, t2)


# ---------------------------------------------------------------------------
# Translation invariance
# ---------------------------------------------------------------------------
def test_translation_invariance():
    task = h_transfer_task()
    pos, pos0 = geometry(task, seed=4)
    base = build(task, pos=pos, pos0=pos0)
    shift = np.array([12.3, -4.5, 7.7])
    moved = build(task, pos=pos + shift, pos0=pos0 + shift)

    node_keep = [i for i in range(NODE_INV_DIM)
                 if i not in NODE_ANCHOR_SLOTS]
    edge_keep = [i for i in range(EDGE_DIM) if i not in EDGE_ANCHOR_SLOTS]
    # non-anchor slots unchanged; r_ij unchanged
    assert np.allclose(moved["h_node_inv"][:, node_keep],
                       base["h_node_inv"][:, node_keep], atol=1e-6)
    assert np.allclose(moved["edge_attr"][:, edge_keep],
                       base["edge_attr"][:, edge_keep], atol=1e-6)
    assert np.allclose(moved["r_ij"], base["r_ij"], atol=1e-6)
    assert np.array_equal(moved["edge_index"], base["edge_index"])
    # anchored quantities are differences => also invariant (documented)
    assert np.allclose(moved["anchor_vec"], base["anchor_vec"], atol=1e-6)
    assert np.allclose(moved["h_node_inv"][:, list(NODE_ANCHOR_SLOTS)],
                       base["h_node_inv"][:, list(NODE_ANCHOR_SLOTS)],
                       atol=1e-6)


# ---------------------------------------------------------------------------
# Collation correctness
# ---------------------------------------------------------------------------
def test_collate_ptr_correctness():
    s1, s2, s3 = build(seed=11), build(seed=12), build(seed=13)
    # enrich with supervision fields the dataset level would attach
    for k, s in enumerate((s1, s2, s3)):
        m = s["key_index"].shape[0]
        s["y_teacher"] = np.full((m, 3), 0.1 * (k + 1), dtype=np.float32)
        s["q_weight"] = np.float32(0.5)
        s["near_ts"] = np.float32(0.0)
        s["terminal"] = np.bool_(False)
        s["capped"] = np.bool_(False)
        s["group_id"] = f"grp{k}"
        s["reaction_id"] = f"rxn{k}"
    b = collate([s1, s2, s3], "cpu", True, EMPTY_VOCAB, EMPTY_VOCAB)

    n1, n2, n3 = (s["h_node_inv"].shape[0] for s in (s1, s2, s3))
    assert b.ptr.tolist() == [0, n1, n1 + n2, n1 + n2 + n3]
    assert b.batch.shape == (n1 + n2 + n3,)
    assert b.batch[:n1].tolist() == [0] * n1
    assert b.batch[n1:n1 + n2].tolist() == [1] * n2

    # global node indices resolve back to per-sample slices
    for bi, s in enumerate((s1, s2, s3)):
        lo, hi = int(b.ptr[bi]), int(b.ptr[bi + 1])
        assert torch.allclose(b.h_node_inv[lo:hi],
                              torch.as_tensor(s["h_node_inv"]))
        # key rows offset into the same global axis
        klo, khi = int(b.key_ptr[bi]), int(b.key_ptr[bi + 1])
        expected = torch.as_tensor(s["key_index"] + lo)
        assert torch.equal(b.key_index[klo:khi], expected)
        # edges of this sample land inside its node range
        n_e = s["edge_index"].shape[1]
        e_per = b.edge_index.shape[1] // 3
        sl = slice(bi * e_per, (bi + 1) * e_per)
        assert torch.all(b.edge_index[0, sl] >= lo)
        assert torch.all(b.edge_index[0, sl] < hi)

    assert b.y_teacher is not None and b.y_teacher.shape == (b.key_index.shape[0], 3)
    assert b.group_ids == ("grp0", "grp1", "grp2")
    assert b.q_weight is not None and b.q_weight.tolist() == [0.5, 0.5, 0.5]


def test_collate_matches_dataclass_fields():
    s = build(seed=5)
    b = collate([s], "cpu", True, EMPTY_VOCAB, EMPTY_VOCAB)
    assert isinstance(b, KeyAtomBatch)
    names = {f.name for f in dataclasses.fields(KeyAtomBatch)}
    assert set(vars(b).keys()) == names


def test_task19_vocab_unk_and_scalars():
    task = make_task((1, 8), key_rows=(0,), bonds_start=[FakeBond(0, 1, 1.0)],
                     bonds_other=[FakeBond(0, 1, 1.0)],
                     charge=-1, multiplicity=2)
    s = build(task, method_hash=stable_id_hash("B3LYP"),
              solvent_hash=stable_id_hash("water"), rho_known=True,
              solvent_known=True)
    vocab_m = build_vocab([stable_id_hash("B3LYP"), stable_id_hash("PBE")])
    vocab_s = build_vocab([stable_id_hash("water")])
    t = build_task19(s, vocab_m, vocab_s)
    assert t[0] == 1.0 and t[1] == 0.0                       # branch 0
    m_idx = vocab_m[stable_id_hash("B3LYP")]
    assert t[2 + m_idx] == 1.0 and t[2:10].sum() == 1.0
    s_idx = vocab_s[stable_id_hash("water")]
    assert t[10 + s_idx] == 1.0 and t[10:14].sum() == 1.0
    assert t[TASK_DIM - 5] == pytest.approx(-1.0 / 4.0)      # charge/4
    assert t[TASK_DIM - 4] == pytest.approx(1.0 / 4.0)       # (mult-1)/4
    assert t[TASK_DIM - 3] == pytest.approx(0.08 / 0.08)     # d0/0.08
    assert t[TASK_DIM - 2] == 1.0 and t[TASK_DIM - 1] == 1.0

    # UNK: unseen hashes land in the FIRST one-hot position
    t2 = build_task19(s, {}, {})
    assert t2[2] == 1.0 and t2[10:14].sum() == 1.0 and t2[10] == 1.0


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------
def test_vocab_unk_and_train_only(tmp_path):
    vocab = build_vocab([7, 3])
    assert vocab == {3: 1, 7: 2}
    assert vocab_index(vocab, 99) == 0                      # UNK
    assert vocab_index(vocab, 7) == 2
    assert METHOD_VOCAB_SIZE == 8 and SOLVENT_VOCAB_SIZE == 4

    from pes2ts_core.pesn.offline.data.key_atom_dataset import build_vocabs

    h_train = stable_id_hash("B3LYP")
    h_hold = stable_id_hash("wB97M")
    root = tmp_path / "features"
    for rid, method in (("rxnA", h_train), ("rxnB", h_hold)):
        d0dir = root / "d0_0.08"
        d0dir.mkdir(parents=True, exist_ok=True)
        np.savez(d0dir / f"{rid}.branch0.npz",
                 method_id_hash=np.array([method], dtype=np.int64),
                 solvent_id_hash=np.array([0], dtype=np.int64))
    m_vocab, s_vocab = build_vocabs(root, ["rxnA"])
    assert h_train in m_vocab
    assert h_hold not in m_vocab                            # train-only


def test_stable_id_hash_stable():
    assert stable_id_hash("B3LYP") == stable_id_hash("B3LYP")
    assert stable_id_hash("B3LYP") != stable_id_hash("PBE")
    assert stable_id_hash("x") >= 0


# ---------------------------------------------------------------------------
# Dataset round-trip (synthetic Lane A seams)
# ---------------------------------------------------------------------------
def test_precompute_dataset_roundtrip(tmp_path):
    from pes2ts_core.pesn.offline.data.key_atom_dataset import (
        V5FeatureDataset,
        build_pools,
        pool_shard_paths,
        precompute_features,
    )

    task = h_transfer_task()
    task.reaction_id = "rxnT"
    task.reaction_group_id = "grpT"
    pos, pos0 = geometry(task, seed=9)
    n_frames, n_key = pos.shape[0], len(task.key_rows)

    # --- synthetic native/teacher/task inputs (Lane A shapes) --------------
    native_root = tmp_path / "native"
    teacher_root = tmp_path / "teacher"
    task_root = tmp_path / "tasks"
    feature_root = tmp_path / "features"
    for d in (native_root, teacher_root / "d0_0.08", task_root):
        d.mkdir(parents=True, exist_ok=True)

    s_frames = 4
    pos_stack = np.stack([pos + 0.01 * f for f in range(s_frames)])
    np.savez(native_root / "rxnT.branch0.npz",
             pos=pos_stack.astype(np.float64),
             key_progress=np.linspace(0.0, 1.0, s_frames),
             q_frame=np.array([1.0, 0.5, 0.25, 0.1]),
             key_rows=np.asarray(task.key_rows, dtype=np.int64))
    (native_root / "rxnT.branch0.meta.json").write_text(
        '{"reference_rows": [1], "reference_weights": [1.0]}')

    start_row = np.array([1, 3], dtype=np.int64)
    np.savez(teacher_root / "d0_0.08" / "rxnT.branch0.npz",
             start_row=start_row,
             u_raw=np.zeros((2, n_key, 3), dtype=np.float32),
             eligible=np.array([True, False]),
             reason_id=np.array([0, 1], dtype=np.uint8),
             near_ts=np.array([0.0, 1.0], dtype=np.float32),
             terminal=np.array([False, True]),
             capped=np.array([False, False]),
             d_eff=np.array([0.08, 0.0]))
    (task_root / "rxnT.task.json").write_text('{"synthetic": true}')

    def fake_loader(path):
        assert Path(path).name == "rxnT.task.json"
        return task

    def fake_align(x0, x_t, rows, weights):
        return x0.copy()          # identity alignment is fine for the test

    config = {"root": str(tmp_path), "native_root": "native",
              "teacher_root": "teacher", "task_root": "tasks",
              "feature_root": "features", "d0_values": [0.08],
              "conflict_weights": {"rxnT": 0.25}}
    result = precompute_features(config, task_loader=fake_loader,
                                 align_fn=fake_align)
    assert result.n_samples == 2 and result.n_eligible == 1
    shard = feature_root / "d0_0.08" / "rxnT.branch0.npz"
    assert shard.is_file()
    assert result.feature_spec_path is not None

    with np.load(shard) as data:
        assert data["sample_ptr"].tolist() == [0, 4, 8]     # 2 frames each
        assert data["key_ptr"].tolist() == [0, n_key, 2 * n_key]
        assert data["q_weight"].tolist() == [0.5 * 0.25, 0.1 * 0.25]
        assert data["eligible"].tolist() == [True, False]
        assert data["y_teacher"].shape == (2 * n_key, 3)
        assert data["h_node_inv"].shape == (8, NODE_INV_DIM)

    # --- lazy dataset + collate -------------------------------------------
    ds = V5FeatureDataset([shard])
    assert len(ds) == 2
    sample = ds[0]
    assert sample["h_node_inv"].shape == (4, NODE_INV_DIM)
    assert sample["edge_index"].shape[1] == sample["edge_attr"].shape[0]
    assert sample["y_teacher"].shape == (n_key, 3)
    assert sample["reaction_id"] == "rxnT"
    vocab_m = build_vocab([stable_id_hash("B3LYP")])
    batch = ds.get_collate_fn(True, vocab_m, {})([ds[0], ds[1]])
    assert isinstance(batch, KeyAtomBatch)
    assert batch.ptr.tolist() == [0, 4, 8]
    assert batch.y_teacher is not None

    # --- pools -------------------------------------------------------------
    pools = build_pools(feature_root, {"grpT": ["rxnT"]}, d0_tag="0.08")
    cycles = pools["grpT"][0]
    assert cycles and all(len(c) <= 16 for c in cycles)
    shard_list = pool_shard_paths(feature_root, "0.08")
    assert len(shard_list) == 1
    # the single eligible row (local idx 0) must appear; q weight rides along
    entries = [e for c in cycles for e in c]
    assert all(e[0] == 0 for e in entries)
    assert all(e[1] == 0 for e in entries)                # only eligible local idx
    assert any(abs(e[2] - 0.125) < 1e-9 for e in entries)  # q_frame 0.5 × 0.25
