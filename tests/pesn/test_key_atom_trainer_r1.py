"""V5.1a-R1 new-loss lane gates (campaign V5_1A_R1_REPAIR).

CPU-fast gates for the R1 additions (protocol
``outputs/pesn/v5_1_r1/protocol_r1.json``, ``new_losses`` section):

* L_strong / L_dead2 bucket semantics through the REAL collate + loss
  path (fixtures mirror ``tests/pesn/test_key_atom_trainer_v51.py``):
  zero-cases, bucket-mean normalization (1 vs 5 strong pairs with
  identical per-pair error), finite gradients w.r.t. ``u``, the frozen
  pair-bucket edge policy (half-open [lo, hi) — a pair exactly on an
  edge belongs to the STRONGER bucket; dead is strictly |dd_true| <
  0.005 A);
* A0-equivalence: with r1 lambdas 0 the loss output is BIT-identical to
  the pre-change function (golden reference captured from the pre-R1
  code on a frozen fixture, plus legacy-config vs explicit-zero-config
  comparison over every term);
* arm resolution from ``config/campaigns/v5_1_r1.yaml`` (A0/A2/A3/A4
  ``r1_strong``/``r1_dead`` flags → configured screening lambdas, off → 0);
* loss-schema version: ``pesn_key_atom_bc_loss_v3`` when either r1
  lambda > 0, unchanged ``pesn_key_atom_bc_loss_v2`` otherwise;
* a tiny SYNTHETIC end-to-end v51 fit per r1 arm (Lane B monkeypatched)
  verifying run_record / metrics / checkpoint arm payloads carry the r1
  lambdas and the v3/v2 schema stamp.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pes2ts_core.config_loader import load_config_mapping
from pes2ts_core.pesn.contracts.key_atom import (
    EDGE_DIM,
    EDGE_SLOTS,
    NODE_INV_DIM,
    TASK_DIM,
    KeyAtomBatch,
)
from pes2ts_core.pesn.models import require_torch
from pes2ts_core.pesn.offline import training as _training_pkg  # noqa: F401
from pes2ts_core.pesn.offline.training import key_atom_bc as kbc

torch = require_torch()
DTYPE = torch.float64

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPO = PROJECT_ROOT

R1_CONFIG = REPO / "config/campaigns/v5_1_r1.yaml"


# ---------------------------------------------------------------------------
# Bucket fixtures: disjoint key pairs along x with EXACT prescribed Δd
# ---------------------------------------------------------------------------

def _bucket_graph(dd_true, dd_pred=None, *, edits=None, rid="RXN_R1",
                  branch: int = 0, n_extra: int = 1) -> KeyAtomBatch:
    """Single-frame graph whose consecutive key pairs realize exact Δd.

    Pair ``i`` connects nodes ``(2i, 2i+1)`` separated by 2⁻¹⁰ A on the
    x-axis; the teacher displacement ``y[2i+1] = (dd_true[i], 0, 0)``
    gives EXACT ``Δd_true_i = ‖r + y_j − y_i‖ − ‖r‖ = dd_true[i]`` bit-for-bit
    (dyadic spacing keeps every intermediate representable), and
    likewise ``u`` for ``dd_pred``.  All nodes are key atoms.
    ``dd_pred`` defaults to a fixed 0.003 A shift.
    """
    dd_true = [float(t) for t in dd_true]
    if dd_pred is None:
        dd_pred = [0.003] * len(dd_true)
    dd_pred = [float(t) for t in dd_pred]
    assert len(dd_true) == len(dd_pred)
    m = len(dd_true)
    n_keys = 2 * m
    assert all(t > -1.4 for t in dd_true + dd_pred), "shrink past bond length"
    # Key spacing 2^-10 A keeps the EXACT-Δd arithmetic at the 1-ulp
    # level: (2^-10 + t) is representable exactly for every binary64 t
    # used below, so exact_delta_d_torch realizes dd == t BIT-exactly —
    # required for the frozen edge-policy assertions (0.02 / 0.005).
    spacing = 2.0 ** -10
    pos = np.zeros((n_keys + n_extra, 3), dtype=np.float64)
    pos[:n_keys, 0] = spacing * np.arange(n_keys)
    # spectator atoms keep the graph non-degenerate without touching keys
    pos[n_keys:, 0] = 6.0 + 0.7 * np.arange(n_extra)
    key_rows = list(range(n_keys))
    pi_rows = [2 * i for i in range(m)]
    pj_rows = [2 * i + 1 for i in range(m)]
    pairs = list(zip(pi_rows, pj_rows))
    edges = []
    for i in range(n_keys + n_extra - 1):
        edges.append((i, i + 1, True))
        edges.append((i + 1, i, True))
    for a, b in pairs:
        edges.append((a, b, True))
        edges.append((b, a, True))
    e_src = [a for a, _b, _c in edges]
    e_dst = [b for _a, b, _c in edges]
    e_attr = np.zeros((len(edges), EDGE_DIM), dtype=np.float64)
    for k, (_a, _b, cov) in enumerate(edges):
        if cov:
            e_attr[k, EDGE_SLOTS["covalent_start"]] = 1.0
            e_attr[k, EDGE_SLOTS["covalent_other"]] = 1.0
    pos0 = pos + 0.1
    anchor = (pos - pos.mean(0)) - (pos0 - pos0.mean(0))
    h_inv = np.zeros((n_keys + n_extra, NODE_INV_DIM), dtype=np.float64)
    h_inv[:, 0] = 1.0
    task = np.zeros(TASK_DIM, dtype=np.float64)
    task[0 if branch == 0 else 1] = 1.0
    y = np.zeros((n_keys, 3), dtype=np.float64)
    u = np.zeros((n_keys, 3), dtype=np.float64)
    for i in range(m):
        y[2 * i + 1, 0] = dd_true[i]
        u[2 * i + 1, 0] = dd_pred[i]
    if edits is None:
        edits = [1.0 if i % 2 == 0 else -1.0 for i in range(m)]
    n = n_keys + n_extra
    return KeyAtomBatch(
        z_atomic=torch.as_tensor(
            np.full(n, 1, dtype=np.int64)),
        h_node_inv=torch.as_tensor(h_inv),
        pos=torch.as_tensor(pos, dtype=DTYPE),
        pos0=torch.as_tensor(pos0, dtype=DTYPE),
        anchor_vec=torch.as_tensor(anchor, dtype=DTYPE),
        edge_index=torch.as_tensor(np.stack([e_src, e_dst]), dtype=torch.int64),
        edge_attr=torch.as_tensor(e_attr),
        r_ij=torch.as_tensor(
            pos[np.asarray(e_src)] - pos[np.asarray(e_dst)], dtype=DTYPE),
        r0_ij=torch.as_tensor(
            pos0[np.asarray(e_src)] - pos0[np.asarray(e_dst)], dtype=DTYPE),
        task19=torch.as_tensor(task).unsqueeze(0),
        batch=torch.zeros(n, dtype=torch.int64),
        ptr=torch.tensor([0, n], dtype=torch.int64),
        key_index=torch.as_tensor(key_rows, dtype=torch.int64),
        key_ptr=torch.tensor([0, n_keys], dtype=torch.int64),
        pair_index=torch.as_tensor(np.stack([pi_rows, pj_rows]),
                                   dtype=torch.int64),
        pair_edit=torch.as_tensor(np.asarray(edits, dtype=np.float64)),
        degK=torch.full((n_keys,), 2.0, dtype=DTYPE),
        y_teacher=torch.as_tensor(y),
        q_weight=torch.ones(1, dtype=DTYPE),
        near_ts=torch.tensor([0.1], dtype=DTYPE),
        terminal=torch.tensor([False]),
        capped=torch.tensor([False]),
        d0=torch.tensor([0.05], dtype=DTYPE),
        group_ids=(rid,),
        reaction_ids=(rid,),
        branch=torch.tensor([branch], dtype=torch.int64),
    )


def _collate_single_graphs(items):
    """Real multi-graph collate (mirrors the v51 trainer fixture)."""
    z, h, pos, pos0, anc, ei, ea, rij, r0ij = [], [], [], [], [], [], [], [], []
    task, batch_vec, ptr = [], [], [0]
    key_index, key_ptr, pi, pj, pe, degk = [], [0], [], [], [], []
    y, qw, near, term, cap, d0s, br = [], [], [], [], [], [], []
    node_off = key_off = 0
    for g, item in enumerate(items):
        n = int(item.pos.shape[0])
        m = int(item.key_index.numel())
        z.append(item.z_atomic)
        h.append(item.h_node_inv)
        pos.append(item.pos)
        pos0.append(item.pos0)
        anc.append(item.anchor_vec)
        ei.append(item.edge_index + node_off)
        ea.append(item.edge_attr)
        rij.append(item.r_ij)
        r0ij.append(item.r0_ij)
        task.append(item.task19[0])
        batch_vec.append(torch.full((n,), g, dtype=torch.int64))
        ptr.append(ptr[-1] + n)
        key_index.append(item.key_index + node_off)
        key_ptr.append(key_ptr[-1] + m)
        pi.append(item.pair_index[0] + node_off)
        pj.append(item.pair_index[1] + node_off)
        pe.append(item.pair_edit)
        degk.append(item.degK)
        y.append(item.y_teacher)
        qw.append(item.q_weight[0])
        near.append(item.near_ts[0])
        term.append(item.terminal[0])
        cap.append(item.capped[0])
        d0s.append(item.d0[0])
        br.append(item.branch[0])
        node_off += n
        key_off += m
    return KeyAtomBatch(
        z_atomic=torch.cat(z), h_node_inv=torch.cat(h),
        pos=torch.cat(pos), pos0=torch.cat(pos0), anchor_vec=torch.cat(anc),
        edge_index=torch.stack([torch.cat([t[0] for t in ei]),
                                torch.cat([t[1] for t in ei])]),
        edge_attr=torch.cat(ea), r_ij=torch.cat(rij), r0_ij=torch.cat(r0ij),
        task19=torch.stack(task), batch=torch.cat(batch_vec),
        ptr=torch.as_tensor(ptr, dtype=torch.int64),
        key_index=torch.cat(key_index),
        key_ptr=torch.as_tensor(key_ptr, dtype=torch.int64),
        pair_index=torch.stack([torch.cat(pi), torch.cat(pj)]),
        pair_edit=torch.cat(pe), degK=torch.cat(degk),
        y_teacher=torch.cat(y), q_weight=torch.stack(qw),
        near_ts=torch.stack(near), terminal=torch.stack(term),
        capped=torch.stack(cap), d0=torch.stack(d0s),
        group_ids=tuple(it.reaction_ids[0] for it in items),
        reaction_ids=tuple(it.reaction_ids[0] for it in items),
        branch=torch.stack(br),
    )


def _r1_out(batch, u=None, *, lambda_strong=0.1, lambda_dead2=0.1,
            pair_loss="legacy_j", dd_pred: float = 0.006):
    """Run the real loss path on a collated batch.

    ``u`` defaults to a prediction with EXACT per-pair Δd_pred =
    ``dd_pred`` (0.006 A — outside the dead zone so L_dead2 > 0 for
    dead-bucket fixtures); pass an explicit tensor for grad tests.
    """
    cfg = kbc.KeyAtomLossConfig(pair_loss=pair_loss)
    if u is None:
        u = torch.zeros_like(batch.y_teacher)
        u[1::2, 0] = dd_pred  # second atom of each disjoint key pair
    return kbc.key_atom_bc_loss_vectorized(
        u, batch.y_teacher, batch.degK, batch.key_ptr, batch.pos,
        batch.key_index, batch.pair_index, config=cfg, lambda_cos=0.0,
        pair_edit=batch.pair_edit, s=None, lambda_strong=lambda_strong,
        lambda_dead2=lambda_dead2)


# ---------------------------------------------------------------------------
# Zero-cases + finite gradients + bucket normalization + edge policy
# ---------------------------------------------------------------------------

def test_r1_zero_case_no_strong_or_dead_pairs():
    """No strong pair ⇒ L_strong == 0 exactly; no dead pair ⇒ L_dead2 == 0."""
    # all |dd_true| in [0.005, 0.02): no strong pair, no dead pair
    mid = _collate_single_graphs([
        _bucket_graph([0.005, 0.008, 0.015, 0.0199])])
    out = _r1_out(mid)
    assert out["n_strong_pairs"] == 0
    assert out["n_dead2_pairs"] == 0
    assert float(out["l_strong"]) == 0.0
    assert float(out["l_dead2"]) == 0.0
    assert float(out["loss"]) >= 0.0 and bool(torch.isfinite(out["loss"]))

    # strongly biased teacher: no dead pair
    strong_only = _collate_single_graphs([
        _bucket_graph([0.03, 0.05, 0.02, 0.1])])
    out_s = _r1_out(strong_only, lambda_dead2=0.1)
    assert out_s["n_strong_pairs"] == 4
    assert out_s["n_dead2_pairs"] == 0
    assert float(out_s["l_dead2"]) == 0.0
    assert float(out_s["l_strong"]) > 0.0

    # tiny teacher: no strong pair
    dead_only = _collate_single_graphs([
        _bucket_graph([0.001, -0.002, 0.0049])])
    out_d = _r1_out(dead_only, lambda_strong=0.1)
    assert out_d["n_strong_pairs"] == 0
    assert float(out_d["l_strong"]) == 0.0
    assert out_d["n_dead2_pairs"] == 3
    assert float(out_d["l_dead2"]) > 0.0

    # zero-edit pairs never enter either bucket
    zedit = _collate_single_graphs([
        _bucket_graph([0.05, 0.001], edits=[0.0, 0.0])])
    out_z = _r1_out(zedit)
    assert out_z["n_strong_pairs"] == 0 and out_z["n_dead2_pairs"] == 0
    assert float(out_z["l_strong"]) == 0.0
    assert float(out_z["l_dead2"]) == 0.0


def test_r1_finite_gradients_wrt_u():
    """Both new terms backprop finite gradients w.r.t. u (leaf tensor)."""
    batch = _collate_single_graphs([
        _bucket_graph([0.03, 0.001, -0.04, 0.006, 0.02, 0.004])])
    u = batch.y_teacher.clone().requires_grad_(True)
    out = _r1_out(batch, u, lambda_strong=0.3, lambda_dead2=0.3)
    assert out["n_strong_pairs"] > 0 and out["n_dead2_pairs"] > 0
    assert bool(torch.isfinite(out["loss"]))
    out["loss"].backward()
    assert u.grad is not None
    assert bool(torch.isfinite(u.grad).all())
    # per-term gradients are finite too
    u2 = batch.y_teacher.clone().requires_grad_(True)
    out2 = _r1_out(batch, u2, lambda_strong=0.3, lambda_dead2=0.3)
    out2["l_strong"].backward(retain_graph=True)
    assert bool(torch.isfinite(u2.grad).all())
    u2.grad = None
    out2["l_dead2"].backward()
    assert bool(torch.isfinite(u2.grad).all())


def test_r1_bucket_mean_normalization():
    """L_strong is a BATCH bucket mean: 1 strong pair == 5 strong pairs
    with identical per-pair error."""
    # per-pair error identical: dd_true = 0.04, dd_pred = 0.01 ⇒ err = -0.03
    one = _collate_single_graphs([_bucket_graph([0.04], [0.01])])
    five = _collate_single_graphs([
        _bucket_graph([0.04] * 5, [0.01] * 5)])
    out_one = _r1_out(one, lambda_strong=0.3, dd_pred=0.01)
    out_five = _r1_out(five, lambda_strong=0.3, dd_pred=0.01)
    assert out_one["n_strong_pairs"] == 1
    assert out_five["n_strong_pairs"] == 5
    v_one = float(out_one["l_strong"])
    v_five = float(out_five["l_strong"])
    assert v_one > 0.0
    assert v_one == pytest.approx(v_five, rel=1e-12, abs=0.0)

    # analytic value: huber(|err|/0.25; δ=0.25) with err = 0.03 →
    # e = 0.12 ≤ 0.25 ⇒ quadratic 0.5 * 0.12² = 0.0072
    assert v_one == pytest.approx(0.5 * (0.03 / 0.25) ** 2, rel=1e-9)

    # dead bucket behaves the same way (bucket mean over the batch)
    d_one = _collate_single_graphs([_bucket_graph([0.002], [0.01])])
    d_five = _collate_single_graphs([_bucket_graph([0.002] * 5, [0.01] * 5)])
    dv_one = float(_r1_out(d_one, dd_pred=0.01)["l_dead2"])
    dv_five = float(_r1_out(d_five, dd_pred=0.01)["l_dead2"])
    assert dv_one > 0.0
    assert dv_one == pytest.approx(dv_five, rel=1e-12, abs=0.0)
    # analytic: max(0, |0.01| − 0.005)² = 0.005² = 2.5e-5
    assert dv_one == pytest.approx((0.01 - 0.005) ** 2, rel=1e-9)


def test_r1_edge_policy_buckets():
    """Frozen edge policy (protocol pair_buckets_angstrom): a pair exactly
    on an edge belongs to the STRONGER bucket; dead is strictly <
    0.005 A.

    The fixture's 2⁻¹⁰ Å dyadic bond makes exact_delta_d_torch realize
    positive dd_true BIT-exactly (verified), so the 0.02 / 0.005 decimal
    edges are tested at the true float64 threshold values — including
    ``nextafter`` complements on both sides.
    """
    import math

    # |dd_true| == 0.02 → strong (inclusive lower edge)
    at_strong = _collate_single_graphs([_bucket_graph([0.02])])
    out = _r1_out(at_strong, lambda_strong=0.1, lambda_dead2=0.1)
    assert out["n_strong_pairs"] == 1
    assert out["n_dead2_pairs"] == 0
    assert float(out["l_strong"]) > 0.0
    # nextafter(0.02, 0) — one ulp BELOW → not strong
    below = _collate_single_graphs([
        _bucket_graph([math.nextafter(0.02, 0.0)])])
    out_b = _r1_out(below, lambda_strong=0.1, lambda_dead2=0.1)
    assert out_b["n_strong_pairs"] == 0
    assert float(out_b["l_strong"]) == 0.0
    # margin-negative shrink (bond crossing compensated): |dd| ≈ 0.048
    neg_strong = _collate_single_graphs([_bucket_graph([-0.05])])
    out_ns = _r1_out(neg_strong, lambda_strong=0.1, lambda_dead2=0.1)
    assert out_ns["n_strong_pairs"] == 1

    # |dd_true| == 0.005 - eps → dead (eps: 1 ulp AND 1e-9 variants)
    for eps_val in (math.nextafter(0.005, 0.0), 0.005 - 1e-9):
        just_below = _collate_single_graphs([_bucket_graph([eps_val])])
        out_j = _r1_out(just_below, lambda_strong=0.1, lambda_dead2=0.1)
        assert out_j["n_dead2_pairs"] == 1, eps_val
        assert out_j["n_strong_pairs"] == 0, eps_val
        assert float(out_j["l_dead2"]) > 0.0

    # |dd_true| == 0.005 exactly → NOT dead (strict <) and not strong
    at_dead = _collate_single_graphs([_bucket_graph([0.005, 0.006])])
    out_d = _r1_out(at_dead, lambda_strong=0.1, lambda_dead2=0.1)
    assert out_d["n_dead2_pairs"] == 0
    assert out_d["n_strong_pairs"] == 0
    assert float(out_d["l_dead2"]) == 0.0
    assert float(out_d["l_strong"]) == 0.0
    # margin-negative dead: |dd| ≈ 4.7e-5 after bond crossing → dead
    neg_dead = _collate_single_graphs([_bucket_graph([-0.002])])
    out_nd = _r1_out(neg_dead, lambda_strong=0.1, lambda_dead2=0.1)
    assert out_nd["n_dead2_pairs"] == 1
    assert out_nd["n_strong_pairs"] == 0


# ---------------------------------------------------------------------------
# A0-equivalence: r1 lambdas 0 ⇒ BIT-identical to the pre-change function
# ---------------------------------------------------------------------------

#: Golden outputs captured from the PRE-R1 ``key_atom_bc_loss_vectorized``
#: on the frozen fixture below (committed verbatim; see git history of
#: this file for the capture).  Every pre-change key must still compare
#: EXACTLY equal when both r1 lambdas are 0.
GOLDEN_PRECHANGE = json.loads(r'''{
 "exact_trio": {
  "l_atom": 0.34848849843659063,
  "l_cos": 1.1461964328065726,
  "l_dead": 0.01185122396622882,
  "l_mag": 0.060723763038945666,
  "l_pair": 0.43323038152409055,
  "l_pair_reg": 0.43323038152409055,
  "l_sign": 11.980006218379076,
  "lambda_cos": 0.02,
  "lambda_dead": 0.1,
  "lambda_mag": 0.5,
  "lambda_sign": 0.2,
  "loss": 3.015575865446678,
  "n_frames": 2,
  "n_pairs_active": 4,
  "pair_loss": "exact_trio"
 },
 "legacy_pair_edit": {
  "l_atom": 0.34848849843659063,
  "l_cos": 1.1461964328065726,
  "l_dead": 0.0,
  "l_mag": 0.060723763038945666,
  "l_pair": 0.4439030338847837,
  "l_pair_reg": 0.4439030338847837,
  "l_sign": 0.0,
  "lambda_cos": 0.02,
  "lambda_dead": 0.1,
  "lambda_mag": 0.5,
  "lambda_sign": 0.2,
  "loss": 0.6237258255545868,
  "n_frames": 2,
  "n_pairs_active": 0,
  "pair_loss": "legacy_j"
 },
 "legacy_plain": {
  "l_atom": 0.34848849843659063,
  "l_cos": 1.1461964328065726,
  "l_dead": 0.0,
  "l_mag": 0.0,
  "l_pair": 0.4439030338847837,
  "l_pair_reg": 0.4439030338847837,
  "l_sign": 0.0,
  "lambda_cos": 0.02,
  "lambda_dead": 0.0,
  "lambda_mag": 0.0,
  "lambda_sign": 0.0,
  "loss": 0.5933639440351139,
  "n_frames": 2,
  "n_pairs_active": 0,
  "pair_loss": "legacy_j"
 }
}''')


def _golden_inputs(seed: int = 101, n_graphs: int = 2, keys_per_graph: int = 3):
    """Frozen fixture (copied verbatim from the V5.1 loss-test builders)."""
    rng = np.random.default_rng(seed)
    key_ptr = [0]
    us, ys, degks, poss, key_rows = [], [], [], [], []
    pi, pj = [], []
    offset_n, key_off = 0, 0
    for g in range(n_graphs):
        n = keys_per_graph + int(rng.integers(2, 5))
        poss.append(rng.uniform(-2, 2, size=(n, 3)))
        m = keys_per_graph
        us.append(rng.normal(scale=0.05, size=(m, 3)))
        ys.append(rng.normal(scale=0.05, size=(m, 3)))
        degks.append(rng.uniform(0.5, 3.0, size=m))
        key_rows.append(np.arange(offset_n, offset_n + m))
        for k in range(m - 1):  # GLOBAL node-row pair space (ADR-0026 §2.6)
            pi.append(key_rows[g][k])
            pj.append(key_rows[g][k + 1])
        key_off += m
        key_ptr.append(key_ptr[-1] + m)
        offset_n += n
    pos = torch.as_tensor(np.concatenate(poss), dtype=DTYPE)
    key_index = torch.as_tensor(np.concatenate(key_rows), dtype=torch.int64)
    pair_index = torch.as_tensor(np.stack([pi, pj]), dtype=torch.int64)
    n_pairs = pair_index.shape[1]
    edits = rng.choice([1.0, -1.0], size=n_pairs)
    y = torch.as_tensor(
        rng.normal(scale=0.05, size=(len(us) * keys_per_graph, 3)),
        dtype=DTYPE)
    u = torch.as_tensor(np.concatenate(us), dtype=DTYPE)
    return {
        "u": u,
        "y": y,
        "degK": torch.as_tensor(np.concatenate(degks), dtype=DTYPE),
        "key_ptr": torch.as_tensor(key_ptr, dtype=torch.int64),
        "pos": pos,
        "key_index": key_index,
        "pair_index": pair_index,
        "pair_edit": torch.as_tensor(edits, dtype=DTYPE),
        "s": u.norm(dim=-1).detach() + 0.02,
    }


def _golden_calls(inputs):
    """The three frozen call shapes — arguments limited to the PRE-R1
    signature so this helper runs identically before and after the change."""
    out = {}
    out["legacy_plain"] = kbc.key_atom_bc_loss_vectorized(
        inputs["u"], inputs["y"], inputs["degK"], inputs["key_ptr"],
        inputs["pos"], inputs["key_index"], inputs["pair_index"],
        config=kbc.KeyAtomLossConfig(), lambda_cos=0.02)
    out["legacy_pair_edit"] = kbc.key_atom_bc_loss_vectorized(
        inputs["u"], inputs["y"], inputs["degK"], inputs["key_ptr"],
        inputs["pos"], inputs["key_index"], inputs["pair_index"],
        config=kbc.KeyAtomLossConfig(), lambda_cos=0.02,
        pair_edit=inputs["pair_edit"], s=inputs["s"],
        lambda_mag=0.5, lambda_sign=0.2, lambda_dead=0.1)
    out["exact_trio"] = kbc.key_atom_bc_loss_vectorized(
        inputs["u"], inputs["y"], inputs["degK"], inputs["key_ptr"],
        inputs["pos"], inputs["key_index"], inputs["pair_index"],
        config=kbc.KeyAtomLossConfig(pair_loss="exact_trio"), lambda_cos=0.02,
        pair_edit=inputs["pair_edit"], s=inputs["s"],
        lambda_mag=0.5, lambda_sign=0.2, lambda_dead=0.1)
    return out


def test_r1_zero_lambdas_bit_identical_to_prechange():
    """A0-equivalence: every pre-change term on the frozen fixture equals
    the committed pre-R1 golden reference EXACTLY (bit-for-bit), for both
    the plain legacy path and the exact_trio path."""
    inputs = _golden_inputs()
    post = _golden_calls(inputs)
    assert set(post) == set(GOLDEN_PRECHANGE)
    r1_extra = {"l_strong", "l_dead2", "lambda_strong", "lambda_dead2",
                "n_strong_pairs", "n_dead2_pairs"}
    for name, out in post.items():
        golden = GOLDEN_PRECHANGE[name]
        # every pre-change key survives
        assert set(golden) <= set(out), (name, set(out) - set(golden))
        assert (set(out) - set(golden)) <= r1_extra, (
            name, set(out) - set(golden))
        for key, expected in golden.items():
            value = out[key]
            if isinstance(expected, float):
                assert float(value) == expected, (name, key, value, expected)
            else:
                assert value == expected, (name, key, value, expected)


def test_r1_zero_lambda_config_equals_legacy_config_path():
    """Config-gating: a config with r1 lambdas EXPLICITLY 0 and a config
    with no r1 keys at all produce identical outputs — total loss and
    every term, bitwise."""
    inputs = _golden_inputs()
    legacy_cfg = kbc.KeyAtomLossConfig.from_campaign_config(
        {"v51": {"ramp": {"start_update": 250, "ramp_updates": 250}}})
    zero_cfg = kbc.KeyAtomLossConfig.from_campaign_config(
        {"v51": {"r1": {"lambda_strong": 0.0, "lambda_dead2": 0.0,
                        "strong_threshold_angstrom": 0.02,
                        "dead_bucket_threshold_angstrom": 0.005,
                        "tau_dead2_angstrom": 0.005,
                        "huber_delta_strong_normalized": 0.25}}})
    assert legacy_cfg.lambda_strong == 0.0 == zero_cfg.lambda_strong
    assert legacy_cfg.lambda_dead2 == 0.0 == zero_cfg.lambda_dead2
    kwargs = dict(config=legacy_cfg, lambda_cos=0.02,
                  pair_edit=inputs["pair_edit"], s=inputs["s"],
                  lambda_mag=0.5, lambda_sign=0.2, lambda_dead=0.1)
    a = kbc.key_atom_bc_loss_vectorized(
        inputs["u"], inputs["y"], inputs["degK"], inputs["key_ptr"],
        inputs["pos"], inputs["key_index"], inputs["pair_index"], **kwargs)
    kwargs_zero = dict(kwargs)
    kwargs_zero["config"] = zero_cfg
    kwargs_zero["lambda_strong"] = 0.0
    kwargs_zero["lambda_dead2"] = 0.0
    b = kbc.key_atom_bc_loss_vectorized(
        inputs["u"], inputs["y"], inputs["degK"], inputs["key_ptr"],
        inputs["pos"], inputs["key_index"], inputs["pair_index"],
        **kwargs_zero)
    assert set(a) == set(b)
    for key in a:
        if torch.is_tensor(a[key]):
            assert torch.equal(a[key], b[key]), key
        else:
            assert a[key] == b[key], key
    # and the pure-zero call (defaults, no kwargs) matches the golden too
    assert float(a["loss"]) == GOLDEN_PRECHANGE["legacy_pair_edit"]["loss"]


# ---------------------------------------------------------------------------
# Config + arm resolution (config/campaigns/v5_1_r1.yaml)
# ---------------------------------------------------------------------------

def test_r1_loss_config_from_campaign_config_precedence():
    base = kbc.KeyAtomLossConfig.from_campaign_config({})
    # defaults: inactive lambdas + frozen protocol thresholds
    assert base.lambda_strong == 0.0
    assert base.lambda_dead2 == 0.0
    assert base.strong_threshold_angstrom == 0.02
    assert base.dead_bucket_threshold_angstrom == 0.005
    assert base.tau_dead2_angstrom == 0.005
    assert base.huber_delta_strong_normalized == 0.25
    # flat top-level spelling (r1_default_* driver keys) works
    flat = kbc.KeyAtomLossConfig.from_campaign_config(
        {"r1_default_lambda_strong": 0.07,
         "r1_default_lambda_dead2": 0.02})
    assert flat.lambda_strong == 0.07
    assert flat.lambda_dead2 == 0.02
    # nested v51.r1 block beats both flat spellings
    nested = kbc.KeyAtomLossConfig.from_campaign_config(
        {"r1_default_lambda_strong": 0.07, "lambda_strong": 0.5,
         "v51": {"r1": {"lambda_strong": 0.3, "lambda_dead2": 0.1,
                        "strong_threshold_angstrom": 0.03,
                        "dead_bucket_threshold_angstrom": 0.004,
                        "tau_dead2_angstrom": 0.006,
                        "huber_delta_strong_normalized": 0.2}}})
    assert nested.lambda_strong == 0.3
    assert nested.lambda_dead2 == 0.1
    assert nested.strong_threshold_angstrom == 0.03
    assert nested.dead_bucket_threshold_angstrom == 0.004
    assert nested.tau_dead2_angstrom == 0.006
    assert nested.huber_delta_strong_normalized == 0.2


def test_r1_arm_resolution_from_new_yaml():
    cfg = load_config_mapping(R1_CONFIG)
    assert cfg.get("schema") == "pesn_v5_1_r1_v1"
    # A0 = pure B0: r1 flags off → lambdas forced to 0
    a0 = kbc.resolve_arm(cfg, "A0")
    assert (a0.arm_id, a0.anchor, a0.d0) == ("A0", "on", 0.05)
    assert a0.magnitude_head is False and a0.pair_loss == "legacy_j"
    assert a0.r1_strong is False and a0.r1_dead is False
    assert a0.lambda_strong == 0.0 and a0.lambda_dead2 == 0.0
    # arm payload stays the LEGACY 5-key block for A0 (bit-identical runs)
    assert set(a0.to_payload()) == {
        "arm_id", "anchor", "d0", "magnitude_head", "pair_loss"}
    # A2 = B0 + strong on
    a2 = kbc.resolve_arm(cfg, "A2")
    assert a2.r1_strong is True and a2.r1_dead is False
    assert a2.lambda_strong == pytest.approx(
        float(cfg["v51"]["r1"]["lambda_strong"]))
    assert a2.lambda_dead2 == 0.0
    assert a2.magnitude_head is False and a2.pair_loss == "legacy_j"
    assert a2.d0 == 0.05 and a2.anchor == "on"
    # A3 = B0 + dead on
    a3 = kbc.resolve_arm(cfg, "A3")
    assert a3.r1_strong is False and a3.r1_dead is True
    assert a3.lambda_strong == 0.0
    assert a3.lambda_dead2 == pytest.approx(
        float(cfg["v51"]["r1"]["lambda_dead2"]))
    # A4 = both on
    a4 = kbc.resolve_arm(cfg, "A4")
    assert a4.r1_strong is True and a4.r1_dead is True
    assert a4.lambda_strong > 0.0 and a4.lambda_dead2 > 0.0
    # r1 lambdas travel in the arm payload ONLY when active
    assert a2.to_payload()["r1_strong"] is True
    assert a2.to_payload()["lambda_strong"] == a2.lambda_strong
    assert a4.to_payload()["r1_dead"] is True
    assert a4.to_payload()["lambda_dead2"] == a4.lambda_dead2


def test_r1_arm_on_without_configured_lambda_stays_zero():
    """arm-level on → use the CONFIGURED screening weight; a config with
    no configured r1 lambda keeps the arm at 0 (strictly config-gated)."""
    cfg = {"d0_angstrom": 0.05,
           "sprint": {"arm_table": {
               "A2": {"d0": 0.05, "anchor": "on", "r1_strong": "on"}}}}
    arm = kbc.resolve_arm(cfg, "A2")
    assert arm.r1_strong is True
    assert arm.lambda_strong == 0.0
    cfg_on = {"d0_angstrom": 0.05,
              "v51": {"r1": {"lambda_strong": 0.1}},
              "sprint": {"arm_table": {
                  "A2": {"d0": 0.05, "anchor": "on", "r1_strong": "on"},
                  "A0": {"d0": 0.05, "anchor": "on",
                         "r1_strong": "off", "r1_dead": "off"}}}}
    assert kbc.resolve_arm(cfg_on, "A2").lambda_strong == 0.1
    off = kbc.resolve_arm(cfg_on, "A0")
    assert off.r1_strong is False and off.lambda_strong == 0.0


def test_r1_arm_flags_validation():
    cfg = {"d0_angstrom": 0.05,
           "v51": {"r1": {"lambda_strong": 0.1}},
           "sprint": {"arm_table": {
               "A2": {"d0": 0.05, "anchor": "on", "r1_strong": "maybe"}}}}
    with pytest.raises(kbc.KeyAtomBCTrainerError) as excinfo:
        kbc.resolve_arm(cfg, "A2")
    assert "ARM_FIELD_INVALID" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Loss schema version (v3 iff either r1 lambda > 0)
# ---------------------------------------------------------------------------

def test_r1_loss_schema_version_gate():
    assert kbc.LOSS_SCHEMA_VERSION == "pesn_key_atom_bc_loss_v2"
    assert kbc.LOSS_SCHEMA_VERSION_R1 == "pesn_key_atom_bc_loss_v3"
    assert kbc.ACCEPTED_CONFIG_SCHEMAS == (
        "pesn_v5_sprint_draft2", "pesn_v5_1_sprint_v1", "pesn_v5_1_r1_v1")
    assert kbc.loss_schema_version(kbc.KeyAtomLossConfig()) == (
        "pesn_key_atom_bc_loss_v2")
    assert kbc.loss_schema_version(
        kbc.KeyAtomLossConfig(lambda_strong=0.03)) == (
            "pesn_key_atom_bc_loss_v3")
    assert kbc.loss_schema_version(
        kbc.KeyAtomLossConfig(lambda_dead2=0.3)) == (
            "pesn_key_atom_bc_loss_v3")
    from_cfg = kbc.KeyAtomLossConfig.from_campaign_config(
        {"v51": {"r1": {"lambda_dead2": 0.1}}})
    assert kbc.loss_schema_version(from_cfg) == "pesn_key_atom_bc_loss_v3"
    zero = kbc.KeyAtomLossConfig.from_campaign_config(
        {"v51": {"r1": {"lambda_strong": 0.0, "lambda_dead2": 0.0}}})
    assert kbc.loss_schema_version(zero) == "pesn_key_atom_bc_loss_v2"


# ---------------------------------------------------------------------------
# Synthetic end-to-end v51 fits for the r1 arms (Lane B monkeypatched)
# ---------------------------------------------------------------------------

def _base_r1_config():
    """Tiny CPU config mirroring v5_1_r1.yaml (schema + r1 block)."""
    return {
        "schema": "pesn_v5_1_r1_v1",
        "d0_angstrom": 0.05,
        "a_ref_angstrom": 0.08,
        "lambda_pair": 0.5,
        "lambda_cos": 0.02,
        "cos_start_update": 250,
        "cos_ramp_updates": 250,
        "tau_norm_angstrom": 0.02,
        "cos_soft_norm_angstrom": 1e-4,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "warmup_updates": 2,
        "min_lr": 1e-4,
        "batch_size": 8,
        "reactions_per_batch": 2,
        "frames_per_reaction": 2,
        "grad_clip_norm": 1.0,
        "eval_every_updates": 2,
        "early_stop_patience_checks": 8,
        "early_stop_min_delta": 0.005,
        "paths": {"split_manifest": "data/manifests/v5/split_manifest.json",
                  "feature_root": "outputs/pesn/v5/features"},
        "sprint": {"campaign_id": "V5_1A_R1_REPAIR", "arm_table": {
            "A0": {"d0": 0.05, "anchor": "on", "magnitude_head": False,
                   "pair_loss": "legacy_j", "r1_strong": False,
                   "r1_dead": False},
            "A2": {"d0": 0.05, "anchor": "on", "magnitude_head": False,
                   "pair_loss": "legacy_j", "r1_strong": True,
                   "r1_dead": False},
            "A3": {"d0": 0.05, "anchor": "on", "magnitude_head": False,
                   "pair_loss": "legacy_j", "r1_strong": False,
                   "r1_dead": True},
        }},
        "v51": {
            "inner_val": {
                "n_val_clusters": 3,
                "seed": 20261019,
                "selection": {"rmse_tolerance": 1.02, "patience_checks": 8,
                              "min_delta": 0.005, "min_update_floor": 500},
            },
            "magnitude_head": {"mode": "per_atom", "lambda_mag": 0.5},
            "pair_exact": {"lambda_sign": 0.2, "lambda_dead": 0.1,
                           "tau_sign_angstrom": 0.005,
                           "tau_dead_angstrom": 0.005,
                           "active_threshold_angstrom": 0.005},
            "ramp": {"start_update": 250, "ramp_updates": 250},
            "r1": {"lambda_strong": 0.1, "lambda_dead2": 0.1,
                   "strong_threshold_angstrom": 0.02,
                   "dead_bucket_threshold_angstrom": 0.005,
                   "tau_dead2_angstrom": 0.005,
                   "huber_delta_strong_normalized": 0.25},
        },
        "device": {"device": "cpu", "require_cuda": False},
    }


def _graph_batch(rng, *, n: int = 9, n_keys: int = 3, rid: str,
                 branch: int, d0: float = 0.05, near: float = 0.1,
                 y_scale: float = 0.08) -> KeyAtomBatch:
    """Copied from the v51 trainer fixture (real-model batches)."""
    pos = rng.uniform(-2.5, 2.5, size=(n, 3))
    key_rows = sorted(rng.choice(n, size=n_keys, replace=False).tolist())
    pairs = [(key_rows[i], key_rows[i + 1]) for i in range(n_keys - 1)]
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1, True))
        edges.append((i + 1, i, True))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) == 1:
                continue
            if float(np.linalg.norm(pos[i] - pos[j])) < 4.5:
                edges.append((i, j, False))
                edges.append((j, i, False))
    for a, b in pairs:
        edges.append((a, b, True))
        edges.append((b, a, True))
    e_src = [a for a, _b, _c in edges]
    e_dst = [b for _a, b, _c in edges]
    e_attr = np.zeros((len(edges), EDGE_DIM), dtype=np.float32)
    for k, (_a, _b, cov) in enumerate(edges):
        if cov:
            e_attr[k, EDGE_SLOTS["covalent_start"]] = 1.0
            e_attr[k, EDGE_SLOTS["covalent_other"]] = 1.0
    pos0 = pos + rng.normal(scale=0.3, size=pos.shape)
    anchor = (pos - pos.mean(0)) - (pos0 - pos0.mean(0))
    h_inv = rng.normal(size=(n, NODE_INV_DIM)).astype(np.float32)
    task = np.zeros(TASK_DIM, dtype=np.float32)
    task[0 if branch == 0 else 1] = 1.0
    task[16] = d0 / 0.08
    y = rng.normal(scale=y_scale, size=(n_keys, 3)).astype(np.float32)
    return KeyAtomBatch(
        z_atomic=torch.as_tensor(
            rng.integers(1, 10, size=n).astype(np.int64)),
        h_node_inv=torch.as_tensor(h_inv),
        pos=torch.as_tensor(pos, dtype=torch.float32),
        pos0=torch.as_tensor(pos0, dtype=torch.float32),
        anchor_vec=torch.as_tensor(anchor, dtype=torch.float32),
        edge_index=torch.as_tensor(np.stack([e_src, e_dst]), dtype=torch.int64),
        edge_attr=torch.as_tensor(e_attr),
        r_ij=torch.as_tensor(
            (pos[np.asarray(e_src)] - pos[np.asarray(e_dst)]),
            dtype=torch.float32),
        r0_ij=torch.as_tensor(
            (pos0[np.asarray(e_src)] - pos0[np.asarray(e_dst)]),
            dtype=torch.float32),
        task19=torch.as_tensor(task).unsqueeze(0),
        batch=torch.zeros(n, dtype=torch.int64),
        ptr=torch.tensor([0, n], dtype=torch.int64),
        key_index=torch.as_tensor(key_rows, dtype=torch.int64),
        key_ptr=torch.tensor([0, n_keys], dtype=torch.int64),
        pair_index=torch.as_tensor(np.stack(pairs), dtype=torch.int64),
        pair_edit=torch.as_tensor(
            np.array([1.0 if i % 2 == 0 else -1.0 for i in range(len(pairs))],
                     dtype=np.float32)),
        degK=torch.full((n_keys,), 2.0, dtype=torch.float32),
        y_teacher=torch.as_tensor(y),
        q_weight=torch.ones(1, dtype=torch.float32),
        near_ts=torch.tensor([near], dtype=torch.float32),
        terminal=torch.tensor([False]),
        capped=torch.tensor([False]),
        d0=torch.tensor([d0], dtype=torch.float32),
        group_ids=(rid,),
        reaction_ids=(rid,),
        branch=torch.tensor([branch], dtype=torch.int64),
    )


@pytest.fixture()
def synthetic_lane(tmp_path, monkeypatch):
    """Monkeypatched Lane B over tiny synthetic graphs (copied from the
    v51 trainer fixture)."""
    rng = np.random.default_rng(20261019)
    samples: dict[str, KeyAtomBatch] = {}
    train: dict[str, dict[int, list[dict]]] = {}
    inner_rows: dict[str, list[dict]] = {}
    for i in range(10):
        rid = f"RXN_T{i:02d}"
        train[rid] = {}
        inner_rows[rid] = []
        for branch in (0, 1):
            rows = []
            for j in range(2):
                sid = f"{rid}.b{branch}.i{j}"
                samples[sid] = _graph_batch(rng, rid=rid, branch=branch)
                rows.append({"sample_id": sid, "q_weight": 1.0,
                             "group_id": rid})
            train[rid][branch] = rows
            inner_rows[rid].extend(rows)
    dev_rows = []
    for branch in (0, 1):
        sid = f"RXN_DEV.b{branch}.i0"
        samples[sid] = _graph_batch(rng, rid="RXN_DEV", branch=branch)
        dev_rows.append({"sample_id": sid, "q_weight": 1.0,
                         "group_id": "RXN_DEV"})
    family_of = {g: f"fam{i % 3}" for i, g in enumerate(train)}
    size_of = {g: (i % 2) + 1 for i, g in enumerate(train)}

    def fake_pools(config, *, fold, d0, split_path, inner_val=None):
        out = {"train": train, "dev_dense": dev_rows}
        if inner_val is not None:
            train_ids, val_ids, split_hash = kbc.inner_val_split(
                sorted(train), family_of=family_of, size_of=size_of,
                n_val=int(inner_val.get("n_val", 3)),
                seed=int(inner_val.get("seed", 20261019)))
            out["inner_train"] = {g: train[g] for g in train_ids}
            out["inner_val_dense"] = [r for g in val_ids
                                      for r in inner_rows[g]]
            out["inner_split"] = {
                "train_ids": list(train_ids), "val_ids": list(val_ids),
                "n_val": int(inner_val.get("n_val", 3)),
                "seed": int(inner_val.get("seed", 20261019)),
                "hash": split_hash,
                "train_rids": list(train_ids), "val_rids": list(val_ids),
                "n_val_clusters": len(val_ids),
            }
        return out

    def fake_view(*, config, samples: list, fold, d0, anchor, split_path,
                  device):
        view = [SAMPLES_BY_ID[row["sample_id"]] for row in samples]
        return view, _collate_single_graphs

    SAMPLES_BY_ID = samples
    monkeypatch.setattr(kbc, "load_lane_b_pools", fake_pools)
    monkeypatch.setattr(kbc, "load_lane_b_view", fake_view)
    return {"samples": samples, "train": train, "dev_rows": dev_rows}


def _make_trainer(tmp_path, *, arm_id: str, updates: int = 4,
                  config=None, run_name: str = "run"):
    cfg = config if config is not None else _base_r1_config()
    arm = kbc.resolve_arm(cfg, arm_id)
    return kbc.KeyAtomBCTrainer(
        config=cfg, config_path=tmp_path / "cfg.yaml", arm=arm, fold=0,
        seed=17, updates=updates, mode="v51", run_dir=tmp_path / run_name,
        device_override="cpu", repo_root=REPO)


def test_r1_v51_fit_end_to_end_synthetic(tmp_path, synthetic_lane):
    """A2 end-to-end: run_record / metrics / checkpoints carry the r1 arm
    block and the v3 loss schema; artifacts land."""
    cfg = _base_r1_config()
    trainer = _make_trainer(tmp_path, arm_id="A2", updates=4, config=cfg)
    assert trainer.loss_config.lambda_strong == 0.1
    assert trainer.loss_config.lambda_dead2 == 0.0
    status = trainer.fit()
    assert status == kbc.STATUS_COMPLETE
    run_dir = trainer.run_dir
    assert (run_dir / "eval.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint_best.pt").is_file()

    record = json.loads((run_dir / "run_record.json").read_text("utf-8"))
    assert record["loss_schema_version"] == "pesn_key_atom_bc_loss_v3"
    assert record["arm"]["arm_id"] == "A2"
    assert record["arm"]["r1_strong"] is True
    assert record["arm"]["r1_dead"] is False
    assert record["arm"]["lambda_strong"] == 0.1
    assert record["arm"]["lambda_dead2"] == 0.0
    metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    assert metrics["loss_schema_version"] == "pesn_key_atom_bc_loss_v3"
    assert metrics["arm"]["lambda_strong"] == 0.1
    import torch as _torch
    best = _torch.load(run_dir / "checkpoints" / "checkpoint_best.pt",
                       map_location="cpu", weights_only=False)
    assert best["arm"]["r1_strong"] is True
    assert best["arm"]["lambda_strong"] == 0.1


def test_r1_a0_fit_reports_v2_schema(tmp_path, synthetic_lane):
    """A0 = pure B0: r1 flags off even though the config carries r1
    lambdas ⇒ arm forces 0 ⇒ loss schema stays v2 and the arm payload
    stays the legacy block."""
    cfg = _base_r1_config()
    trainer = _make_trainer(tmp_path, arm_id="A0", updates=4, config=cfg,
                            run_name="run_a0")
    assert trainer.loss_config.lambda_strong == 0.0
    assert trainer.loss_config.lambda_dead2 == 0.0
    assert trainer.fit() == kbc.STATUS_COMPLETE
    record = json.loads(
        (trainer.run_dir / "run_record.json").read_text("utf-8"))
    assert record["loss_schema_version"] == "pesn_key_atom_bc_loss_v2"
    assert set(record["arm"]) == {
        "arm_id", "anchor", "d0", "magnitude_head", "pair_loss"}
    metrics = json.loads(
        (trainer.run_dir / "metrics.json").read_text("utf-8"))
    assert metrics["loss_schema_version"] == "pesn_key_atom_bc_loss_v2"


def test_r1_config_schema_accepted_by_trainer(tmp_path, synthetic_lane):
    """The new yaml schema passes the trainer schema gate; the trainer
    resolves the A3 arm flags from it."""
    cfg = _base_r1_config()
    trainer = _make_trainer(tmp_path, arm_id="A3", updates=2, config=cfg,
                            run_name="run_a3")
    assert trainer.config["schema"] == "pesn_v5_1_r1_v1"
    assert trainer.arm.r1_dead is True
    assert trainer.loss_config.lambda_dead2 == 0.1
    assert trainer.loss_config.lambda_strong == 0.0
    assert kbc.loss_schema_version(trainer.loss_config) == (
        "pesn_key_atom_bc_loss_v3")
