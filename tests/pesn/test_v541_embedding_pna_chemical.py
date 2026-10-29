"""V5.4.1 WP3b — E2 ``pna_chemical`` candidate tests (plan §6.7 E2,
§6.6/§6.8 rows 3/4).

Covered here (candidate-level; the model-level battery lives in
``test_v541_embedding_chemical_model.py``):

* frozen §6.7 aggregation — hand-computed small graphs for EVERY
  aggregator × scaler block (mean/max/min/std × identity/
  amplification/attenuation) with ``mu_log = ln 2`` (so d=1 scalers are
  exactly 1 and d=3 gives amp=2, att=½), zero-neighbour nodes (all
  aggregates ZERO, self preserved), single-neighbour (population std
  exactly 0) and finite gradients through the guarded std;
* degree scalers: d=0 ⇒ (1, 1, 1); amp·att = 1 for d > 0; population
  (not unbiased) std;
* the FROZEN stats intake — ``set_degree_stats(mu_log, stats_sha256)``
  with typed ``V541_DEGREE_STATS_INVALID`` on bad input, typed
  ``V541_DEGREE_STATS_MISSING`` on forward-before-set, buffer/
  provenance checkpoint round-trips and typed refusal of a tampered
  checkpoint; forward NEVER fits stats nor reads a split;
* §6.8 row 2 direction/duplicate discipline, E(3)/permutation
  invariance, F2 discipline and the diagnostics switch.

Registry discipline: the E2 module is imported INSIDE an autouse fixture
(its import registers ``E2``) and the registry is restored afterwards.
"""

from __future__ import annotations

import math

import pytest
import torch

from pes2ts_core.pesn.models.v541.embedding_registry import (
    build_embedding,
    embedding_spec,
    is_embedding_implemented,
    registry_restore,
    registry_snapshot,
)
from pes2ts_core.pesn.models.v541.feature_providers import build_chem_graph
from pes2ts_core.pesn.models.v541.interfaces import (
    HIDDEN_V541,
    normalize_branch_output,
)
from tests.pesn.test_v541_embedding_chemical_fixtures import (
    assert_all_floats,
    assert_close,
    bonded_pair_batch,
    chain_batch,
    chemical_batch_matrix,
    controlled_pna,
    duplicate_first_edge,
    flip_edge_directions,
    load_e2,
    patch_count_only_helpers,
    poison_first_edge_attr,
    random_h,
    sn2_bidirectional_batch,
    sn2_forward_batch,
    single_atom_batch,
    sphere_center_batch,
)
from tests.pesn.test_v541_feature_providers_fixtures import (
    forward_batch,
    no_neighbor_batch,
    permute_forward_batch,
    random_rotation,
    reflection_matrix,
    transform_forward_batch,
)

SEED = 13
#: frozen test digest (64 hex chars) used by set_degree_stats
SHA_OK = "ab" * 32
#: mu_log = ln 2 makes d=1 scalers exactly (1, 1, 1) and d=3 (2, ½)
MU_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def e2():
    """Import (⇒ register E2) inside the test, restore afterwards."""
    snap = registry_snapshot()
    module = load_e2()
    yield module
    registry_restore(snap)


def _ready(module) -> "object":
    """A fresh E2 with frozen stats (mu_log = ln2)."""
    mod = module.PNAChemicalEmbedding()
    mod.set_degree_stats(MU_LN2, SHA_OK)
    return mod


def _select_block_readout(mod, block: str) -> None:
    controlled_pna(mod, block=block)


# ---------------------------------------------------------------------------
# construction / registry contract
# ---------------------------------------------------------------------------
def test_e2_shapes_registry_and_frozen_towers(e2) -> None:
    mod = e2.PNAChemicalEmbedding()
    assert mod.width == HIDDEN_V541 == 48
    assert e2.PNA_TOWERS == 1 and e2.AGGREGATORS == ("mean", "max", "min", "std")
    assert e2.SCALERS == ("identity", "amplification", "attenuation")
    # P_e: 8 → 48; msg MLP: 144 → 48 → 48; readout: 624 → 48 → 48
    assert tuple(mod.pe_proj.weight.shape) == (48, 8)
    assert tuple(mod.msg_mlp[0].weight.shape) == (48, 144)
    assert tuple(mod.msg_mlp[2].weight.shape) == (48, 48)
    assert tuple(mod.readout[0].weight.shape) == (48, 624)   # 13·48
    assert tuple(mod.readout[2].weight.shape) == (48, 48)
    assert isinstance(mod.msg_mlp[1], torch.nn.ReLU)
    assert isinstance(mod.readout[1], torch.nn.ReLU)
    # no dropout / BatchNorm / pooling (plan §6.7 公共项)
    forbidden = (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                 torch.nn.AvgPool1d, torch.nn.MaxPool1d, torch.nn.AvgPool2d,
                 torch.nn.MaxPool2d)
    assert not any(isinstance(m, forbidden) for m in mod.modules())
    # param count: P_e 9w + msg 4w²+2w + readout 14w²+2w = 18w² + 13w
    w = HIDDEN_V541
    assert sum(p.numel() for p in mod.parameters()) == 18 * w * w + 13 * w
    # the import registered the REAL E2 builder (WP3 path)
    assert is_embedding_implemented("E2") is True
    built = build_embedding("E2")
    assert isinstance(built, e2.PNAChemicalEmbedding)
    spec = embedding_spec("E2")
    assert spec.family == "pna_chemical"
    assert spec.output == "residual_node_delta"
    assert spec.in_main_matrix is True
    # frozen stats surface: unset until set_degree_stats
    assert mod.degree_stats_state == "unset"
    assert mod.last_diagnostics == {}
    # width parameter free for E5-style reuse
    narrow = e2.PNAChemicalEmbedding(width=24)
    assert narrow.width == 24
    assert tuple(narrow.readout[0].weight.shape) == (24, 13 * 24)


# ---------------------------------------------------------------------------
# frozen stats intake (set_degree_stats / checkpoint / typed refusals)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_mu", [0.0, -1.0, float("nan"), float("inf")])
def test_set_degree_stats_rejects_bad_mu(e2, bad_mu) -> None:
    mod = e2.PNAChemicalEmbedding()
    with pytest.raises(Exception) as exc:
        mod.set_degree_stats(bad_mu, SHA_OK)
    assert getattr(exc.value, "code", "") == "V541_DEGREE_STATS_INVALID"
    assert mod.degree_stats_state == "unset"          # untouched


@pytest.mark.parametrize("bad_sha", ["abc", "zz" * 32, b"short", [0] * 31])
def test_set_degree_stats_rejects_bad_sha(e2, bad_sha) -> None:
    mod = e2.PNAChemicalEmbedding()
    with pytest.raises(Exception) as exc:
        mod.set_degree_stats(MU_LN2, bad_sha)
    assert getattr(exc.value, "code", "") == "V541_DEGREE_STATS_INVALID"
    assert mod.degree_stats_state == "unset"


def test_forward_before_set_degree_stats_is_missing(e2) -> None:
    batch = bonded_pair_batch()
    mod = e2.PNAChemicalEmbedding()
    h = random_h(2, SEED)
    with pytest.raises(Exception) as exc:
        mod(h, batch)
    assert getattr(exc.value, "code", "") == "V541_DEGREE_STATS_MISSING"
    # a refused call never populates diagnostics
    assert mod.last_diagnostics == {}


def test_set_degree_stats_overwrites_and_provenance(e2) -> None:
    mod = e2.PNAChemicalEmbedding()
    mod.set_degree_stats(MU_LN2, SHA_OK)
    assert mod.degree_stats_state == "ready"
    prov = mod.degree_stats_provenance
    assert prov["state"] == "ready"
    assert abs(float(prov["mu_log"]) - MU_LN2) < 1e-6
    assert prov["stats_sha256"] == SHA_OK
    # a SECOND (different) valid set wins — last write, like any buffer
    other = "cd" * 32
    mod.set_degree_stats(0.5, other)
    assert mod.degree_stats_provenance["stats_sha256"] == other
    assert mod.degree_stats_provenance["mu_log"] == 0.5
    # an INVALID re-set raises and leaves the previous stats in place
    with pytest.raises(Exception):
        mod.set_degree_stats(0.0, SHA_OK)
    assert mod.degree_stats_state == "ready"
    assert mod.degree_stats_provenance["stats_sha256"] == other


def test_stats_buffers_checkpoint_roundtrip(e2) -> None:
    """mu_log AND its provenance digest are BUFFERS: a plain state_dict
    round-trip re-arms the module (plan §7.3 checkpoint content)."""
    batch = bonded_pair_batch()
    h = random_h(2, SEED)
    src = _ready(e2)
    sd = src.state_dict()
    assert "mu_log" in sd and "stats_sha256" in sd and "stats_ready" in sd
    dst = e2.PNAChemicalEmbedding()
    assert dst.degree_stats_state == "unset"
    dst.load_state_dict(sd)
    assert dst.degree_stats_state == "ready"
    assert dst.degree_stats_provenance["stats_sha256"] == SHA_OK
    with torch.no_grad():
        a = src(h, batch)
        b = dst(h, batch)
    assert torch.equal(a, b)                       # bitwise re-armed


def test_tampered_checkpoint_is_invalid(e2) -> None:
    """ready=1 but mu_log=0 (a hand-edited checkpoint) → INVALID."""
    batch = bonded_pair_batch()
    h = random_h(2, SEED)
    sd = _ready(e2).state_dict()
    sd["mu_log"] = torch.tensor(0.0)
    mod = e2.PNAChemicalEmbedding()
    mod.load_state_dict(sd)
    assert mod.degree_stats_state == "invalid"
    with pytest.raises(Exception) as exc:
        mod(h, batch)
    assert getattr(exc.value, "code", "") == "V541_DEGREE_STATS_INVALID"


def test_forward_never_fits_stats_or_reads_split(
        monkeypatch: pytest.MonkeyPatch, e2) -> None:
    """Forward must NEVER recompute the stats (plan §6.6: "只算一次并
    在构造后读取, 不在 forward 中重算") nor read a split."""
    import pes2ts_core.pesn.offline.training.v541_degree_stats as dmod

    def boom(*_args, **_kwargs):
        raise AssertionError("degree stats were recomputed in forward")

    monkeypatch.setattr(dmod, "compute_degree_stats", boom)
    mod = _ready(e2)                       # stats arrive via the buffer
    batch = bonded_pair_batch()
    with torch.no_grad():
        out = mod(random_h(2, SEED), batch)  # must not raise
    assert bool(torch.isfinite(out).all())
    # the consumed mu_log is exactly the frozen one (not a refit)
    assert mod.degree_stats_provenance["mu_log"] == mod._mu_log_value


# ---------------------------------------------------------------------------
# degree scalers (§6.7): identity / amplification / attenuation
# ---------------------------------------------------------------------------
def test_degree_scaler_rules(e2) -> None:
    mod = e2.PNAChemicalEmbedding()
    degree = torch.tensor([0.0, 1.0, 2.0, 3.0])
    identity, amp, att = mod._degree_scalers(degree, math.log(2.0))
    assert torch.equal(identity, torch.ones_like(degree))
    # d = 0 → both scalars explicitly 1 (§6.7 d=0 rule)
    assert float(amp[0]) == 1.0 and float(att[0]) == 1.0
    # d = 1 with mu_log = ln2 → (1, 1) exactly
    assert abs(float(amp[1]) - 1.0) < 1e-6
    assert abs(float(att[1]) - 1.0) < 1e-6
    # amplification = log1p(d)/mu, attenuation its reciprocal
    assert abs(float(amp[2]) - math.log(3.0) / math.log(2.0)) < 1e-6
    assert abs(float(amp[3]) - 2.0) < 1e-6                 # ln4/ln2 = 2
    assert abs(float(att[3]) - 0.5) < 1e-6
    assert torch.allclose(amp * att, torch.ones_like(degree), atol=1e-6)
    # all values finite for any positive mu
    assert bool(torch.isfinite(amp).all() and torch.isfinite(att).all())


def test_population_std_is_population(e2) -> None:
    """std uses the POPULATION variance (÷n, not ÷n−1) — §6.7."""
    # two messages [1, 3]: population std = 1, unbiased would be √2
    total = torch.tensor([[4.0]])
    total_sq = torch.tensor([[10.0]])
    count = torch.tensor([2])
    std = e2._population_std(total, total_sq, count)
    assert float(std.item()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# hand-computed aggregation blocks (§6.7 / §6.8 row 3)
# ---------------------------------------------------------------------------
def test_hand_computed_bonded_pair_blocks(e2) -> None:
    """d = 1 everywhere, mu_log = ln2 ⇒ all scalers are 1, std = 0."""
    batch = bonded_pair_batch(1.0)
    h = random_h(2, SEED)
    m = h[0] + h[1]                       # single message per node
    for block, expected in (
            ("meanxidentity", m), ("maxxidentity", m), ("minxidentity", m),
            ("stdxidentity", torch.zeros_like(m)),
            ("meanxamplification", m), ("meanxattenuation", m)):
        mod = _ready(e2)
        _select_block_readout(mod, block)
        with torch.no_grad():
            out = mod(h, batch)
        assert_close(out[0], expected, f"bonded block {block}")
        assert_close(out[1], expected, f"bonded block {block} (node1)")


def test_hand_computed_shell_blocks(e2) -> None:
    """Centre with 3 neighbours (d=3): amp = 2, att = ½, std of the
    three messages, max/min elementwise — all hand-computable."""
    batch = sphere_center_batch([1.0, 1.0, 1.0])
    h = random_h(4, SEED)
    msgs = torch.stack([h[0] + h[1], h[0] + h[2], h[0] + h[3]])   # [3, 48]
    mean_c = msgs.mean(dim=0)
    max_c, _ = msgs.max(dim=0)
    min_c, _ = msgs.min(dim=0)
    std_c = msgs.std(dim=0, unbiased=False)
    leaf_msg = h[1] + h[0]
    cases = (
        ("meanxidentity", mean_c, leaf_msg),
        ("maxxidentity", max_c, leaf_msg),
        ("minxidentity", min_c, leaf_msg),
        ("stdxidentity", std_c, torch.zeros_like(leaf_msg)),
        # d=3, mu=ln2 → amp = ln4/ln2 = 2, att = ½
        ("meanxamplification", 2.0 * mean_c, leaf_msg),
        ("meanxattenuation", 0.5 * mean_c, leaf_msg),
        # d=1 leaf → every scaler is 1
        ("maxxamplification", 2.0 * max_c, leaf_msg),
    )
    for block, centre_expected, leaf_expected in cases:
        mod = _ready(e2)
        _select_block_readout(mod, block)
        with torch.no_grad():
            out = mod(h, batch)
        assert_close(out[0], centre_expected, f"shell block {block}")
        assert_close(out[1], leaf_expected, f"shell block {block} (leaf)")


def test_zero_neighbour_all_stats_zero_self_preserved(e2) -> None:
    """Zero-neighbour nodes: every aggregator block is EXACTLY 0 and the
    self feature survives (plan §6.7: 零邻居统计全零且保留 self)."""
    for batch, n_nodes in ((single_atom_batch(), 1), (no_neighbor_batch(), 2)):
        h = random_h(n_nodes, SEED)
        for block in ("meanxidentity", "maxxattenuation", "minxamplification",
                      "stdxidentity"):
            mod = _ready(e2)
            _select_block_readout(mod, block)
            with torch.no_grad():
                out = mod(h, batch)
            assert float(out.abs().max()) == 0.0, block
        # the SELF block is preserved (positive h ⇒ ReLU inert)
        mod = _ready(e2)
        _select_block_readout(mod, "self")
        with torch.no_grad():
            out = mod(h, batch)
        assert_close(out, torch.relu(h), "zero-neighbour self block")
        diag = mod.last_diagnostics
        assert diag["n_edges"] == 0.0
        assert diag["zero_neighbor"] == float(n_nodes)
        assert diag["zero_neighbor_frac"] == 1.0


def test_single_neighbour_std_is_exactly_zero(e2) -> None:
    """Single-neighbour nodes: population std is EXACTLY 0 (no fp dust)
    and the gradient through the guarded sqrt stays finite."""
    batch = bonded_pair_batch(1.0)
    mod = _ready(e2)
    _select_block_readout(mod, "stdxidentity")
    h = (random_h(2, SEED)).requires_grad_(True)
    out = mod(h, batch)
    assert float(out.detach().abs().max()) == 0.0
    out.sum().backward()
    assert h.grad is not None
    assert bool(torch.isfinite(h.grad).all())
    for pname, p in mod.named_parameters():
        assert p.grad is not None, pname
        assert bool(torch.isfinite(p.grad).all()), pname


def test_chain_hand_values(e2) -> None:
    """3-atom chain: degrees (1, 2, 1) — the middle node aggregates two
    messages, hand-checkable with the controlled message MLP."""
    batch = chain_batch()
    h = random_h(3, SEED)
    mod = _ready(e2)
    _select_block_readout(mod, "meanxidentity")
    with torch.no_grad():
        out = mod(h, batch)
    # messages into1: from 0 and from 2 (receiver-first convention)
    m_from_0 = h[1] + h[0]
    m_from_2 = h[1] + h[2]
    assert_close(out[0], h[0] + h[1], "chain node0")
    assert_close(out[1], 0.5 * (m_from_0 + m_from_2), "chain node1 mean")
    assert_close(out[2], h[2] + h[1], "chain node2")


# ---------------------------------------------------------------------------
# §6.8 row 2 + E(3)/permutation + F2 discipline
# ---------------------------------------------------------------------------
def test_direction_flip_and_duplicates(e2) -> None:
    """Direction flip → BITWISE-equal delta; equal duplicates merge;
    an inconsistent duplicate is a typed provider refusal."""
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = _ready(e2)
    with torch.no_grad():
        base = mod(h, batch)
        flipped = mod(h, flip_edge_directions(batch))
        merged = mod(h, duplicate_first_edge(batch))
    assert torch.equal(base, flipped)
    assert torch.equal(base, merged)
    with pytest.raises(Exception) as exc:
        mod(h, poison_first_edge_attr(batch))
    assert getattr(exc.value, "code", "") == "V541_PROVIDER_EDGE_ATTR_MISMATCH"


def test_view_directions_carry_identical_signed_attrs(e2) -> None:
    batch = sn2_forward_batch()
    view = build_chem_graph(batch)
    assert torch.equal(view.edge_attr[0::2], view.edge_attr[1::2])
    assert float(view.pair_attr[:, 2].abs().max()) > 0.0   # real edit


def test_translation_rotation_reflection_invariance(e2) -> None:
    """E2 reads NO geometry: the delta is bitwise pos-invariant."""
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = _ready(e2)
    with torch.no_grad():
        ref = mod(h, batch)
        shifted = mod(h, transform_forward_batch(
            batch, translate=torch.tensor([12.5, -3.25, 7.75])))
        rotated = mod(h, transform_forward_batch(
            batch, rotate=random_rotation(0)))
        reflected = mod(h, transform_forward_batch(
            batch, rotate=reflection_matrix(2)))
        both = mod(h, transform_forward_batch(
            batch, rotate=random_rotation(3) @ reflection_matrix(0),
            translate=torch.tensor([1.0, 2.0, 3.0])))
    for label, moved in (("translation", shifted), ("rotation", rotated),
                         ("reflection", reflected), ("rot+refl+trans", both)):
        assert torch.equal(ref, moved), f"E2 {label} must be bitwise"


def test_permutation_equivariance(e2) -> None:
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    gen = torch.Generator().manual_seed(17)
    perm = torch.randperm(n, generator=gen)
    permuted = permute_forward_batch(batch, perm)
    mod = _ready(e2)
    with torch.no_grad():
        ref = mod(h, batch)
        out = mod(h[perm], permuted)
    assert_close(ref[perm], out, "E2 permutation")


def test_f2_masked_slots_never_read_and_input_untouched(e2) -> None:
    batch = sn2_forward_batch()
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = _ready(e2)
    before = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
              for k, v in batch.items()}
    with torch.no_grad():
        base = mod(h, batch)
        poisoned = dict(batch)
        attr = batch["edge_attr"].clone()
        attr[:, :12] = 1234.5
        poisoned["edge_attr"] = attr
        out = mod(h, poisoned)
    assert torch.equal(base, out)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, before[key]), key


# ---------------------------------------------------------------------------
# gradients over the fixture matrix
# ---------------------------------------------------------------------------
def test_gradients_finite_over_matrix(e2) -> None:
    for name, batch in chemical_batch_matrix().items():
        n = int(batch["pos"].shape[0])
        h = random_h(n, SEED).requires_grad_(True)
        mod = _ready(e2)
        out = mod(h, batch)
        assert bool(torch.isfinite(out).all()), name
        out.sum().backward()
        assert h.grad is not None and bool(torch.isfinite(h.grad).all()), name
        for pname, p in mod.named_parameters():
            assert p.grad is not None, f"{name}:{pname}"
            assert bool(torch.isfinite(p.grad).all()), f"{name}:{pname}"
        normalize_branch_output(
            out.detach(), n_nodes=n, hidden=HIDDEN_V541,
            ref_dtype=torch.float32)


def test_independent_vs_mixed_batch(e2) -> None:
    from scripts.agent_ops.feature_gate.v3_fixtures import make_sn2_sample

    batch_a = forward_batch([make_sn2_sample("forward")])
    batch_b = forward_batch([make_sn2_sample("reverse", jitter_seed=7)])
    batch_ab = forward_batch([
        make_sn2_sample("forward"), make_sn2_sample("reverse", jitter_seed=7)])
    n_a = int(batch_a["pos"].shape[0])
    n_b = int(batch_b["pos"].shape[0])
    torch.manual_seed(SEED)
    h_a = torch.randn(n_a, 48)
    h_b = torch.randn(n_b, 48)
    mod = _ready(e2)
    with torch.no_grad():
        alone = mod(h_a, batch_a)
        alone_b = mod(h_b, batch_b)
        mixed = mod(torch.cat([h_a, h_b]), batch_ab)
    assert int(batch_ab["graph_index"][:n_a].max()) == 0
    assert_close(alone, mixed[:n_a], "E2 independent-vs-mixed (A)")
    assert_close(alone_b, mixed[n_a:n_a + n_b], "E2 independent-vs-mixed (B)")


# ---------------------------------------------------------------------------
# diagnostics switch (WP3a pattern)
# ---------------------------------------------------------------------------
def test_diagnostics_false_skips_count_only_helpers(
        monkeypatch: pytest.MonkeyPatch, e2) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    cheap = e2.PNAChemicalEmbedding(diagnostics_enabled=False)
    cheap.set_degree_stats(MU_LN2, SHA_OK)
    patch_count_only_helpers(monkeypatch)          # helpers now RAISE
    with torch.no_grad():
        out = cheap(h, batch)                      # must not raise
    assert bool(torch.isfinite(out).all())
    assert cheap.last_diagnostics == {}

    loud = e2.PNAChemicalEmbedding(diagnostics_enabled=True)
    loud.set_degree_stats(MU_LN2, SHA_OK)
    with pytest.raises(AssertionError):
        with torch.no_grad():
            loud(h, batch)


def test_diagnostics_off_keeps_features_bitwise_identical(e2) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    on = _ready(e2)
    off = e2.PNAChemicalEmbedding(diagnostics_enabled=False)
    off.load_state_dict(on.state_dict())
    with torch.no_grad():
        a = on(h, batch)
        b = off(h, batch)
    assert torch.equal(a, b)
    assert on.last_diagnostics and off.last_diagnostics == {}


def test_diagnostics_are_detached_floats_and_copied(e2) -> None:
    batch = sn2_bidirectional_batch(n_graphs=2)
    n = int(batch["pos"].shape[0])
    h = random_h(n, SEED)
    mod = _ready(e2)
    with torch.no_grad():
        mod(h, batch)
    diag = mod.last_diagnostics
    assert_all_floats(diag, "E2.last_diagnostics")
    for key in ("n_nodes", "n_edges", "zero_neighbor", "zero_neighbor_frac",
                "mu_log"):
        assert key in diag, key
    assert diag["n_atoms"] == float(n)
    assert diag["n_chem_edges"] == diag["n_edges"]
    assert abs(diag["mu_log"] - MU_LN2) < 1e-6
    snapshot = mod.last_diagnostics
    snapshot["n_edges"] = -1.0
    assert mod.last_diagnostics["n_edges"] != -1.0
    mod.diagnostics_enabled = False
    with torch.no_grad():
        mod(h, batch)
    assert mod.last_diagnostics == {}
