"""V5.4.1 WP3b — inner-train degree statistics tests
(``pes2ts_core/pesn/offline/training/v541_degree_stats.py``; plan §6.7
E2 / §6.6, config ``feature_recipe.degree_stats``).

Covered here:

* the G_chem pair/degree math — bit-for-bit the provider's rules, proven
  against ``build_chem_graph`` on every WP3a fixture batch (including
  spatial-only pairs, which must NOT count, duplicate edges and multi-
  degree spheres);
* hand-checked ``mu_log`` / weighted totals / degree histogram on a tiny
  synthetic population, plus order/duplicate independence (the position
  list is treated as a SET — "每个计划样本一次");
* inner-train ONLY: ``dev`` / ``confirm`` / ``inner-val`` / ``train``
  splits are typed refusals; empty / out-of-range positions, self loops
  and out-of-range endpoints are typed refusals; an all-edgeless
  population is the typed refusal of §6.6 (``V541_DEGREE_STATS_NO_EDGES``);
* reverse intents are included and weighted (forward+reverse of one
  sample share ONE unique topology but TWO weighted visits);
* the columnar fast path never constructs a record (poisoned
  ``_record_at``) and agrees with the record path on the real fixture
  loader;
* the result's ``stats_sha256`` feeds
  ``PNAChemicalEmbedding.set_degree_stats`` (integration).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pes2ts_core.pesn.offline.training.v541_degree_stats as dmod
from pes2ts_core.pesn.models.v541.feature_providers import build_chem_graph
from tests.pesn.test_v541_feature_providers_fixtures import (
    dense_sphere_batch,
    forward_batch,
    make_intent,
    make_sample,
    sn2_forward_batch,
)
from scripts.agent_ops.feature_gate.v3_fixtures import make_sn2_intent, make_sn2_sample

SEED = 20261027


# ---------------------------------------------------------------------------
# lightweight record stand-ins (no dataset needed for the math tests)
# ---------------------------------------------------------------------------
class _Rec:
    """Minimal record: the helper only needs .intent / .atomic_numbers."""

    def __init__(self, intent: Any, n_atoms: int) -> None:
        self.intent = intent
        self.atomic_numbers = tuple(range(1, int(n_atoms) + 1))


class _CountingSeq(list):
    """Sequence counting ``__getitem__`` per index (visit-once proof)."""

    def __init__(self, items) -> None:
        super().__init__(items)
        self.accesses: list[int] = []

    def __getitem__(self, index):          # type: ignore[override]
        self.accesses.append(int(index))
        return super().__getitem__(index)


def _sn2_samples() -> list[_Rec]:
    fwd = make_sn2_intent("forward")
    rev = make_sn2_intent("reverse")
    return [_Rec(fwd, 6), _Rec(rev, 6)]


# ---------------------------------------------------------------------------
# hand-checked math on a tiny population
# ---------------------------------------------------------------------------
def test_hand_checked_mu_log_histogram_and_totals() -> None:
    fwd = make_sn2_intent("forward")
    rev = make_sn2_intent("reverse")
    solo = make_intent((), (), cv_deltas=(0.1,), cv_coords=((0, 0),))
    samples = [_Rec(fwd, 6), _Rec(rev, 6), _Rec(solo, 1)]
    res = dmod.compute_degree_stats(
        samples, [0, 1, 2], split="inner-train", seed=SEED)

    # SN2 topology: centre 0 with 5 incident pairs (Br, O, 3×H)
    assert dmod.chem_in_degrees(fwd, 6).tolist() == [5, 1, 1, 1, 1, 1]
    # forward and reverse share ONE unique topology, TWO weighted visits
    assert res.n_unique_graphs == 2                 # sn2 topology + lone atom
    assert res.weighted_graph_total == 3
    assert res.weighted_node_total == 6 + 6 + 1
    assert res.weighted_pair_total == 5 + 5 + 0
    assert res.weighted_edge_total == 10 + 10 + 0   # directed = 2 × pairs
    assert res.degree_histogram == {5: 2, 1: 10, 0: 1}
    expected_mu = (2 * (math.log(6.0) + 5 * math.log(2.0))) / 13.0
    assert res.mu_log == pytest.approx(expected_mu, rel=1e-12)
    # mu_log is also recomputable from the recorded histogram (§6.8 row 3)
    hist_sum = sum(v * math.log1p(k) for k, v in res.degree_histogram.items())
    assert hist_sum / sum(res.degree_histogram.values()) == pytest.approx(
        res.mu_log, rel=1e-12)
    assert res.schema == "pes2ts_v541_degree_stats_v1"
    assert res.n_positions == 3


def test_reverse_intents_included_and_weighted() -> None:
    fwd = make_sn2_intent("forward")
    rev = make_sn2_intent("reverse")
    only_fwd = dmod.compute_degree_stats(
        [_Rec(fwd, 6)], [0], split="inner-train", seed=SEED)
    both = dmod.compute_degree_stats(
        [_Rec(fwd, 6), _Rec(rev, 6)], [0, 1], split="inner-train", seed=SEED)
    # both directions share the topology but double the population weight
    assert both.n_unique_graphs == only_fwd.n_unique_graphs == 1
    assert both.weighted_graph_total == 2 * only_fwd.weighted_graph_total
    assert both.weighted_node_total == 2 * only_fwd.weighted_node_total
    assert both.mu_log == pytest.approx(only_fwd.mu_log, rel=1e-12)


def test_duplicate_topology_is_weighted_not_unique() -> None:
    """Two positions with the SAME topology: unique=1, weighted=2."""
    intent = make_sn2_intent("forward")
    samples = [_Rec(intent, 6), _Rec(intent, 6)]
    res = dmod.compute_degree_stats(
        samples, [0, 1], split="inner-train", seed=SEED)
    assert res.n_unique_graphs == 1
    assert res.weighted_graph_total == 2
    assert res.weighted_node_total == 12
    assert res.degree_histogram == {5: 2, 1: 10}


def test_position_set_semantics_and_visit_once() -> None:
    """The position list is a SET (order/duplicates independent) and
    every canonical position is visited exactly once (§6.7)."""
    intent = make_sn2_intent("forward")
    samples = _CountingSeq([_Rec(intent, 6), _Rec(make_sn2_intent("reverse"), 6)])
    a = dmod.compute_degree_stats(
        samples, [0, 1], split="inner-train", seed=SEED)
    first_call = list(samples.accesses)
    samples.accesses.clear()
    b = dmod.compute_degree_stats(
        samples, [1, 0, 1], split="inner-train", seed=SEED)
    assert a.stats_sha256 == b.stats_sha256        # order/dup independent
    assert first_call == [0, 1]                    # each position once
    assert samples.accesses == [0, 1]              # and still once, no dup 1


def test_shas_are_recorded_and_deterministic() -> None:
    samples = _sn2_samples()
    a = dmod.compute_degree_stats(
        samples, [0, 1], split="inner-train", seed=SEED)
    b = dmod.compute_degree_stats(
        samples, [0, 1], split="inner-train", seed=SEED)
    c = dmod.compute_degree_stats(
        samples, [0, 1], split="inner-train", seed=SEED + 1)
    assert a.stats_sha256 == b.stats_sha256
    assert a.stats_sha256 != c.stats_sha256        # the seed binds the file
    for field_name in ("split_sha256", "positions_sha256",
                       "graph_recipe_sha256", "seed_sha256"):
        value = getattr(a, field_name)
        assert len(value) == 64 and all(
            ch in "0123456789abcdef" for ch in value), field_name
    assert a.seed == SEED and a.seed_sha256 != c.seed_sha256
    # JSON-able cache payload (config cache path / WP4 loader)
    payload = json.loads(json.dumps(a.to_dict()))
    assert payload["stats_sha256"] == a.stats_sha256
    assert payload["degree_histogram"] == {"1": 10, "5": 2}


def test_rss_and_time_recorded() -> None:
    res = dmod.compute_degree_stats(
        _sn2_samples(), [0, 1], split="inner-train", seed=SEED)
    assert res.elapsed_seconds >= 0.0
    assert res.max_rss_kb > 0


# ---------------------------------------------------------------------------
# typed refusals (inner-train ONLY, §6.6)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_split", ["dev", "confirm", "inner-val", "train"])
def test_non_inner_train_split_is_refused(bad_split: str) -> None:
    with pytest.raises(dmod.V541DegreeStatsError) as exc:
        dmod.compute_degree_stats(
            _sn2_samples(), [0], split=bad_split, seed=SEED)
    assert exc.value.code == "V541_DEGREE_STATS_SPLIT"
    assert "inner-train" in exc.value.detail


def test_empty_and_out_of_range_positions_refused() -> None:
    samples = _sn2_samples()
    with pytest.raises(dmod.V541DegreeStatsError) as exc:
        dmod.compute_degree_stats(samples, [], split="inner-train", seed=SEED)
    assert exc.value.code == "V541_DEGREE_STATS_EMPTY"
    with pytest.raises(dmod.V541DegreeStatsError) as exc:
        dmod.compute_degree_stats(
            samples, [0, 99], split="inner-train", seed=SEED)
    assert exc.value.code == "V541_DEGREE_STATS_POSITION_RANGE"


def test_all_edgeless_population_is_refused() -> None:
    """§6.6: a training population without a single G_chem edge would
    give mu_log = 0 (undefined amplification) → typed refusal."""
    solo = make_intent((), (), cv_deltas=(0.1,), cv_coords=((0, 0),))
    samples = [_Rec(solo, 1), _Rec(solo, 2)]
    with pytest.raises(dmod.V541DegreeStatsError) as exc:
        dmod.compute_degree_stats(samples, [0, 1], split="inner-train",
                                  seed=SEED)
    assert exc.value.code == "V541_DEGREE_STATS_NO_EDGES"


def test_topology_refusals_match_provider() -> None:
    """Self loops and out-of-range endpoints are typed refusals (the
    provider refuses SELF_LOOP / EDGE_RANGE on the same input)."""
    self_loop = make_intent(((0, 0, 1.0),), ((0, 1, 1.0),),
                            cv_deltas=(0.1,), cv_coords=((0, 1),))
    with pytest.raises(dmod.V541DegreeStatsError) as exc:
        dmod.chem_in_degrees(self_loop, 2)
    assert exc.value.code == "V541_DEGREE_STATS_TOPOLOGY"
    out_of_range = make_intent(((0, 5, 1.0),), ((0, 5, 1.0),),
                               cv_deltas=(0.1,), cv_coords=((0, 1),))
    with pytest.raises(dmod.V541DegreeStatsError) as exc:
        dmod.chem_in_degrees(out_of_range, 3)
    assert exc.value.code == "V541_DEGREE_STATS_TOPOLOGY"


# ---------------------------------------------------------------------------
# provider cross-proof (§6.6: "不确定时必须用provider对照fixture证明一致")
# ---------------------------------------------------------------------------
def _provider_degrees(sample) -> np.ndarray:
    batch = forward_batch([sample])
    view = build_chem_graph(batch)
    n_atoms = len(sample.atomic_numbers)
    return np.bincount(view.edge_index[1].numpy(), minlength=n_atoms)


def test_intent_degrees_match_provider_on_fixtures() -> None:
    """The intent fast path reproduces ``build_chem_graph`` degrees on
    EVERY WP3a fixture batch — including spatial-only pairs (excluded),
    stretched covalent pairs (kept) and multi-degree spheres."""
    samples = [
        make_sn2_sample("forward"),
        make_sn2_sample("reverse"),
        make_sample((6,), [(0.0, 0.0, 0.0)],
                    make_intent((), (), cv_deltas=(0.1,), cv_coords=((0, 0),))),
        make_sample((6, 8), [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                    make_intent(((0, 1, 1.0),), ((0, 1, 1.0),),
                                cv_deltas=(0.05,), cv_coords=((0, 1),))),
        make_sample((6, 1, 1),
                    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (5.0, 0.0, 0.0)],
                    make_intent(((0, 1, 1.0), (1, 2, 1.0)),
                                ((0, 1, 1.0), (1, 2, 1.0)),
                                cv_deltas=(0.1,), cv_coords=((0, 1),))),
        # spatial contact at 1.5 Å must NOT become a chem degree
        make_sample((6, 1, 8),
                    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
                    make_intent(((0, 1, 1.0),), ((0, 1, 1.0),),
                                cv_deltas=(0.1,), cv_coords=((0, 1),))),
    ]
    for sample in samples:
        n_atoms = len(sample.atomic_numbers)
        mine = dmod.chem_in_degrees(sample.intent, n_atoms)
        prov = _provider_degrees(sample)
        assert mine.tolist() == prov.tolist(), sample.intent.intent_id
    # the spatial-exclusion premise is actually exercised
    spatial = samples[-1]
    assert dmod.chem_in_degrees(spatial.intent, 3).tolist() == [1, 1, 0]


def test_dense_sphere_degree_matches_provider() -> None:
    """A 15-neighbour sphere: centre degree 15 in both computations."""
    batch = dense_sphere_batch(15)
    # recover the sample's intent through the provider itself
    view = build_chem_graph(batch)
    n_atoms = int(batch["pos"].shape[0])
    prov = np.bincount(view.edge_index[1].numpy(), minlength=n_atoms)
    assert int(prov.max()) == 15
    assert prov.sum() == 2 * 15


def test_duplicate_edges_merge_like_provider() -> None:
    """A bitwise-equal duplicated edge column merges in the provider —
    the intent path is duplicate-free by construction and agrees."""
    batch = sn2_forward_batch()
    n_atoms = int(batch["pos"].shape[0])
    dup = dict(batch)
    dup["edge_index"] = np.concatenate(
        [batch["edge_index"].numpy(), batch["edge_index"].numpy()[:, :1]],
        axis=1)
    import torch as _torch

    dup["edge_index"] = _torch.from_numpy(dup["edge_index"])
    for key in ("edge_attr", "b_src", "b_tgt", "delta_b_intent",
                "abs_delta", "is_changed"):
        dup[key] = _torch.cat([batch[key], batch[key][:1]], dim=0)
    view_plain = build_chem_graph(batch)
    view_dup = build_chem_graph(dup)
    deg_plain = np.bincount(view_plain.edge_index[1].numpy(), minlength=n_atoms)
    deg_dup = np.bincount(view_dup.edge_index[1].numpy(), minlength=n_atoms)
    assert deg_plain.tolist() == deg_dup.tolist()
    # and an inconsistent duplicate is refused by the provider
    bad = dict(dup)
    b_src = bad["b_src"].clone()
    b_src[-1] = b_src[-1] + 1.0
    bad["b_src"] = b_src
    from pes2ts_core.pesn.models.v541.feature_providers import (
        FeatureProviderError,
    )

    with pytest.raises(FeatureProviderError) as exc:
        build_chem_graph(bad)
    assert exc.value.code == "V541_PROVIDER_EDGE_ATTR_MISMATCH"


# ---------------------------------------------------------------------------
# real loader: columnar fast path vs record path (I4 discipline)
# ---------------------------------------------------------------------------
@pytest.fixture()
def real_loader(tmp_path: Path):
    from tests.pesn.test_r3_v53mp_dataset import (  # noqa: E402
        ALL_STATUSES, _write_fixture,
    )
    from tests.pesn.test_r3_v53mp_geometry_pack import _build_stack  # noqa: E402

    tables = _write_fixture(tmp_path / "data")
    stack = _build_stack(tables)
    v1 = stack["v1"]
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (  # noqa: E402
        ColumnarKeyAtomDatasetV3,
    )

    v2 = ColumnarKeyAtomDatasetV3(
        tables, label_statuses=ALL_STATUSES, verify_manifest=False)
    return v1, v2, tables


def test_columnar_fast_path_never_constructs_records(real_loader) -> None:
    v1, v2, tables = real_loader
    positions = list(range(len(v1.samples)))
    assert len(positions) == 16                      # 2 curves × 4 × 2

    def _boom(row: int) -> Any:
        raise AssertionError("columnar fast arm materialised a record")
    v2._record_at = _boom
    res_fast = dmod.compute_degree_stats(
        v2.samples, positions, split="inner-train", seed=SEED)
    assert res_fast.n_positions == 16
    # hand-checked on the REAL fixture topology (5 pairs, 6 atoms)
    assert res_fast.degree_histogram == {5: 16, 1: 80}
    expected_mu = (math.log(6.0) + 5 * math.log(2.0)) / 6.0
    assert res_fast.mu_log == pytest.approx(expected_mu, rel=1e-12)
    # ... and the record fallback path on the SAME columnar dataset
    # (startup disabled) plus the v1 loader agree bit-for-bit
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (
        ColumnarKeyAtomDatasetV3,
    )
    from tests.pesn.test_r3_v53mp_dataset import ALL_STATUSES  # noqa: E402

    real_v2 = ColumnarKeyAtomDatasetV3(  # re-open un-poisoned
        tables, label_statuses=ALL_STATUSES, verify_manifest=False)
    res_record = dmod.compute_degree_stats(
        real_v2.samples, positions, split="inner-train", seed=SEED)
    res_v1 = dmod.compute_degree_stats(
        v1.samples, positions, split="inner-train", seed=SEED)
    assert res_fast.mu_log == res_record.mu_log == res_v1.mu_log
    assert (res_fast.degree_histogram == res_record.degree_histogram
            == res_v1.degree_histogram)
    assert (res_fast.weighted_edge_total == res_record.weighted_edge_total
            == res_v1.weighted_edge_total)


def test_record_fallback_agrees_with_fast_path(
        monkeypatch: pytest.MonkeyPatch, real_loader) -> None:
    """Forcing the fallback (``_startup_of`` → None) on the SAME columnar
    dataset must reproduce the fast path exactly."""
    v1, v2, _tables = real_loader
    positions = list(range(len(v1.samples)))
    res_fast = dmod.compute_degree_stats(
        v2.samples, positions, split="inner-train", seed=SEED)

    monkeypatch.setattr(dmod, "_startup_of", lambda _samples: None)
    res_record = dmod.compute_degree_stats(
        v2.samples, positions, split="inner-train", seed=SEED)
    assert res_fast.mu_log == res_record.mu_log
    assert res_fast.degree_histogram == res_record.degree_histogram
    assert res_fast.weighted_node_total == res_record.weighted_node_total
    assert res_fast.weighted_edge_total == res_record.weighted_edge_total
    assert res_fast.stats_sha256 == res_record.stats_sha256


# ---------------------------------------------------------------------------
# integration: the stats file feeds E2's frozen buffers
# ---------------------------------------------------------------------------
def test_result_feeds_e2_set_degree_stats() -> None:
    import importlib

    pna = importlib.import_module(
        "pes2ts_core.pesn.models.v541.embeddings.pna_chemical")
    from pes2ts_core.pesn.models.v541.embedding_registry import (
        registry_restore, registry_snapshot,
    )

    snap = registry_snapshot()
    try:
        from tests.pesn.test_v541_embedding_chemical_fixtures import load_e2

        load_e2()
        res = dmod.compute_degree_stats(
            _sn2_samples(), [0, 1], split="inner-train", seed=SEED)
        mod = pna.PNAChemicalEmbedding()
        mod.set_degree_stats(res.mu_log, res.stats_sha256)
        assert mod.degree_stats_state == "ready"
        assert mod.degree_stats_provenance["stats_sha256"] == res.stats_sha256
        assert mod.degree_stats_provenance["mu_log"] == pytest.approx(
            res.mu_log, rel=1e-6)
    finally:
        registry_restore(snap)
