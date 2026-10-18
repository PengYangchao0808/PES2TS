"""Dual-TS feature v4 (E1/E2) tests (synthetic only, no real data).

Covers plan v4 §6 / contract §5 E1/E2 / ADR-0024 item 4:

* 7-channel gradient-augmented edge features — hand-computed analytic values;
* zero-gradient-norm atoms are zeroed and reported via the returned mask;
* ``pesn_dual_ts_features_v4`` spec digest is stable and DIFFERS from the v3
  digest, with ``endpoint_edge_feat == 7`` and ``parent_spec_digest`` bound;
* key-pair whitelist contains ONLY changed-bond union edges;
* ``KeyPairConditioner`` is exchange-symmetric (A/B swap → identical output)
  and pair-endpoint swap invariant, with parameters counted.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from pes2ts_core.pesn.contracts.dual_ts import EndpointBond, EndpointGraph
from pes2ts_core.pesn.features.dual_ts_graph_v4 import (
    DUAL_TS_FEATURES_V4_VERSION,
    KEY_PAIR_RECORD_DIM,
    N_EDGE_FEATS_V4,
    KeyPairConditioner,
    augment_edge_features_gradient,
    build_features_spec_v4,
    build_key_pair_records,
    load_spec_v4_payload,
    select_key_pairs,
)
from pes2ts_core.pesn.features.dual_ts_spec import (
    build_features_spec,
    save_spec_payload,
)

TOL = 1e-12


def _graph(z, bonds) -> EndpointGraph:
    n = int(np.asarray(z).size)
    return EndpointGraph(
        atomic_numbers=np.asarray(z, dtype=int),
        atom_map_ids=tuple(f"m{k + 1:04d}" for k in range(n)),
        bonds=tuple(bonds),
    )


# ---------------------------------------------------------------------------
# (1) 7-channel augmentation — hand-computed example
# ---------------------------------------------------------------------------

def test_augment_edge_features_gradient_hand_computed():
    # atom0 at origin, atom1 at (1,0,0); g0 = (1,0,0), g1 = (0,1,0) (unit).
    x = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    g = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    edge_index = np.asarray([[0, 1], [1, 0]])          # directed both ways
    edge_feats = np.asarray([[1.0, 1.5, 0.0, 0.0],
                             [1.0, 1.5, 0.0, 0.0]])    # identifiable 4ch

    aug, zero_mask = augment_edge_features_gradient(edge_feats, x, g, edge_index)

    assert aug.shape == (2, N_EDGE_FEATS_V4)
    # ghat = g (unit) for both atoms.
    # edge 0 (0->1): rhat_01 = (-1,0,0) → rhat·ghat0 = -1, rhat·ghat1 = 0,
    #                 ghat0·ghat1 = 0.
    assert np.allclose(aug[0, :4], edge_feats[0], atol=TOL)
    assert aug[0, 4] == pytest.approx(-1.0, abs=TOL)
    assert aug[0, 5] == pytest.approx(0.0, abs=TOL)
    assert aug[0, 6] == pytest.approx(0.0, abs=TOL)
    # edge 1 (1->0): rhat_10 = (1,0,0) → rhat·ghat1 = 0, rhat·ghat0 = 1,
    #                 ghat1·ghat0 = 0.
    assert np.allclose(aug[1, :4], edge_feats[1], atol=TOL)
    assert aug[1, 4] == pytest.approx(0.0, abs=TOL)
    assert aug[1, 5] == pytest.approx(1.0, abs=TOL)
    assert aug[1, 6] == pytest.approx(0.0, abs=TOL)
    # no zero-norm atoms here.
    assert zero_mask.tolist() == [False, False]


def test_augment_zero_norm_atoms_zeroed_and_masked():
    x = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    g = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    edge_index = np.asarray([[0, 1, 0], [1, 0, 2]])     # three directed edges
    edge_feats = np.ones((3, 4))

    aug, zero_mask = augment_edge_features_gradient(edge_feats, x, g, edge_index)

    assert zero_mask.tolist() == [False, True, False]
    # atom1's ghat is zeroed → channels touching ghat1 are exactly 0.
    # edge (0->1): rhat·ghat0 = -1 (non-zero), rhat·ghat1 = 0, ghat0·ghat1 = 0.
    assert aug[0, 4] == pytest.approx(-1.0, abs=TOL)
    assert aug[0, 5] == pytest.approx(0.0, abs=TOL)
    assert aug[0, 6] == pytest.approx(0.0, abs=TOL)
    # edge (1->0): rhat·ghat1 = 0, rhat·ghat0 = 1, ghat1·ghat0 = 0.
    assert aug[1, 4] == pytest.approx(0.0, abs=TOL)
    assert aug[1, 5] == pytest.approx(1.0, abs=TOL)
    assert aug[1, 6] == pytest.approx(0.0, abs=TOL)


# ---------------------------------------------------------------------------
# (2) spec v4 digest stable + != v3 + parent binding
# ---------------------------------------------------------------------------

SPEC_KW = dict(
    s_g=0.0511, s_delta_e=0.42, element_vocab=[1, 6, 7, 8],
    population={"wave": "fit", "reactions": 1, "branches": 2, "states": 10},
    n_atoms_range=(3, 3),
)


def test_spec_v4_digest_stable_and_differs_from_v3(tmp_path):
    assert DUAL_TS_FEATURES_V4_VERSION == "pesn_dual_ts_features_v4"
    v3_spec, _ = build_features_spec(**SPEC_KW)
    spec1, payload1 = build_features_spec_v4(**SPEC_KW)
    spec2, payload2 = build_features_spec_v4(**SPEC_KW)

    # stable across builds.
    assert spec1.content_digest() == spec2.content_digest()
    assert payload1 == payload2
    # distinct from v3, parent bound.
    assert spec1.content_digest() != v3_spec.content_digest()
    assert payload1["parent_spec_digest"] == v3_spec.content_digest()
    # layout: edge feat 4 -> 7, node scalars unchanged at 5.
    assert payload1["declared_layout"]["endpoint_edge_feat"] == 7
    assert payload1["declared_layout"]["node_scalar"] == 5
    assert payload1["input_dim"] == sum(payload1["declared_layout"].values())

    # save / load round-trip with digest verification.
    path = tmp_path / "spec_v4.json"
    assert save_spec_payload(payload1, path) == spec1.content_digest()
    loaded, lp = load_spec_v4_payload(path)
    assert loaded.content_digest() == spec1.content_digest()
    assert lp == payload1

    # tampering is caught.
    bad = dict(payload1)
    bad["declared_layout"] = dict(payload1["declared_layout"],
                                  endpoint_edge_feat=8)
    (tmp_path / "bad.json").write_text(
        json.dumps({"spec": bad, "spec_digest": spec1.content_digest()}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_spec_v4_payload(tmp_path / "bad.json")


# ---------------------------------------------------------------------------
# (3) key-pair whitelist: changed-bond union edges only
# ---------------------------------------------------------------------------

def test_key_pair_whitelist_only_contains_changed():
    # (2,3) is an unchanged spectator bond; (0,1) order-changed, (0,2) formed,
    # (1,2) broken.
    a = _graph([6, 1, 7, 8], [EndpointBond(0, 1, 1.0), EndpointBond(1, 2, 1.0),
                              EndpointBond(2, 3, 1.0)])
    b = _graph([6, 1, 7, 8], [EndpointBond(0, 1, 2.0), EndpointBond(0, 2, 1.0),
                              EndpointBond(2, 3, 1.0)])
    key = select_key_pairs(a, b)
    assert {tuple(p) for p in key} == {(0, 1), (0, 2), (1, 2)}
    # unchanged graph pair -> empty whitelist.
    assert select_key_pairs(a, a).shape == (0, 2)


def test_key_pair_records_shape_and_dim():
    a = _graph([6, 1, 7, 8], [EndpointBond(0, 1, 1.0), EndpointBond(1, 2, 1.0)])
    b = _graph([6, 1, 7, 8], [EndpointBond(0, 1, 2.0), EndpointBond(0, 2, 1.0)])
    key = select_key_pairs(a, b)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 3))
    rec = build_key_pair_records(key_pairs=key, x=x, topology_a=a, topology_b=b)
    assert rec.shape == (int(key.shape[0]), KEY_PAIR_RECORD_DIM)


# ---------------------------------------------------------------------------
# (4) KeyPairConditioner: exchange-symmetric + pair-swap invariant + params
# ---------------------------------------------------------------------------

def test_key_pair_conditioner_swap_endpoint_symmetry():
    z = [6, 1, 7, 8]
    a = _graph(z, [EndpointBond(0, 1, 1.0), EndpointBond(1, 2, 1.0)])
    b = _graph(z, [EndpointBond(0, 1, 2.0), EndpointBond(0, 2, 1.0)])
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 3))

    key_ab = select_key_pairs(a, b)
    key_ba = select_key_pairs(b, a)
    rec_ab = build_key_pair_records(key_pairs=key_ab, x=x, topology_a=a, topology_b=b)
    rec_ba = build_key_pair_records(key_pairs=key_ba, x=x, topology_a=b, topology_b=a)
    # exchange-symmetric whitelist and records.
    assert np.array_equal(key_ab, key_ba)
    assert np.allclose(rec_ab, rec_ba, atol=TOL)

    torch.manual_seed(3)
    cond = KeyPairConditioner(hidden=8, pair_hidden=16).to(torch.float64)
    h = torch.randn(4, 8, dtype=torch.float64)
    ki = torch.tensor(key_ab, dtype=torch.long)
    rec = torch.tensor(rec_ab, dtype=torch.float64)

    with torch.no_grad():
        out_ab = cond(h, rec, ki)
        out_ba = cond(h, torch.tensor(rec_ba), ki)             # identical records
        out_swap = cond(h, rec, ki[:, [1, 0]])                 # pair endpoint swap
    assert torch.allclose(out_ab, out_ba, atol=1e-12)
    assert torch.allclose(out_ab, out_swap, atol=1e-12)
    # modulation actually changes h.
    assert not torch.allclose(out_ab, h, atol=1e-12)
    # parameters counted; empty whitelist -> identity map.
    assert cond.n_parameters == sum(p.numel() for p in cond.parameters())
    empty = torch.zeros(0, KEY_PAIR_RECORD_DIM, dtype=torch.float64)
    empty_idx = torch.zeros(0, 2, dtype=torch.long)
    assert torch.allclose(cond(h, empty, empty_idx), h, atol=0.0)
