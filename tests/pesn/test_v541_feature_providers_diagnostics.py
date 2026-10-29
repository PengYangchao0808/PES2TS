"""V5.4.1 WP3a — ``diagnostics=True/False`` equivalence tests.

Bounded per-update cost fix: every public ``build_*`` entry (and the
internal ``build_*_view`` / ``read_forward_batch`` path) accepts
``diagnostics: bool = True``.  Default ``True`` keeps the frozen
behaviour (``counts`` filled with detached Python floats at the cost of
GPU→host reductions).  ``False`` must:

* return ``counts == {}`` on every view;
* keep ALL numerical features BITWISE identical (shape, dtype and row
  order included) and gradients bitwise identical;
* skip every count-only host sync/statistic — proven by monkeypatching
  the count-only helpers (:func:`...common.count_graphs`,
  :func:`...common.diagnostic_sum`) to raise;
* keep every legality check running (cross-graph, duplicate-attribute
  mismatch, non-finite geometry, zero-length angle sides) — turning
  diagnostics off must never weaken validation.

WP3b passes the flag explicitly (monitor steps keep ``True``; the
per-update path uses ``False``).
"""

from __future__ import annotations

import pytest
import torch

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    EDGE_ABS_DELTA_KEY,
    EDGE_ATTR_KEY,
    EDGE_B_SRC_KEY,
    EDGE_B_TGT_KEY,
    EDGE_DELTA_B_KEY,
    EDGE_INDEX_KEY,
    EDGE_IS_CHANGED_KEY,
    NODE_POS_KEY,
)
from pes2ts_core.pesn.models.v541.feature_providers import (
    FeatureProviderError,
    build_angle_triplets,
    build_chem_graph,
    build_legal_neighbor_views,
    build_spatial_graph,
)
from pes2ts_core.pesn.models.v541.feature_providers import (
    angle_triplets as angle_module,
)
from pes2ts_core.pesn.models.v541.feature_providers import (
    common as provider_common,
)
from pes2ts_core.pesn.models.v541.feature_providers import (
    legal_neighbors as legal_module,
)
from pes2ts_core.pesn.models.v541.feature_providers.common import (
    read_forward_batch,
)
from pes2ts_core.pesn.models.v541.feature_providers.legal_neighbors import (
    build_chem_view,
    build_spatial_view,
)
from tests.pesn.test_v541_feature_providers_fixtures import (
    angle_geometry_batch,
    dense_sphere_batch,
    forward_batch,
    make_intent,
    make_sample,
    single_atom_batch,
    sn2_bidirectional_batch,
    sn2_forward_batch,
    sphere_center_batch,
    stretched_bond_batch,
)


def _batches() -> dict[str, dict]:
    """Batches that exercise every count path (graph count, chem
    synthesis count, spatial over-cap/beyond-cutoff, angle boundary and
    cosine diagnostics)."""
    spatial_only = forward_batch([make_sample(
        (6, 8), [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
        make_intent((), (), cv_deltas=(0.05,), cv_coords=((0, 1),)))])
    return {
        "sn2": sn2_forward_batch(),
        "mixed": sn2_bidirectional_batch(n_graphs=2),
        "dense": dense_sphere_batch(15),
        "shell_tie": sphere_center_batch([1.0] * 11 + [2.0] * 3),
        "angle_pi": angle_geometry_batch("pi"),
        "stretched": stretched_bond_batch(5.0),
        "single": single_atom_batch(),
        "spatial_only": spatial_only,
    }


def _tensor_fields(view: object) -> dict[str, torch.Tensor]:
    return {k: v for k, v in vars(view).items()
            if isinstance(v, torch.Tensor)}


def _assert_tensors_identical(a: object, b: object, label: str) -> None:
    """Every tensor field bitwise equal (shape + dtype + value)."""
    fields_a = _tensor_fields(a)
    fields_b = _tensor_fields(b)
    assert set(fields_a) == set(fields_b), label
    for key in sorted(fields_a):
        left, right = fields_a[key], fields_b[key]
        assert left.shape == right.shape, f"{label}:{key} shape"
        assert left.dtype == right.dtype, f"{label}:{key} dtype"
        assert torch.equal(left, right), f"{label}:{key} not bitwise equal"


def _assert_identical(on: object, off: object, label: str) -> None:
    """Tensor equivalence + the flag contract: ``True`` fills counts,
    ``False`` returns an empty dict."""
    _assert_tensors_identical(on, off, label)
    assert on.counts, f"{label}: diagnostics=True must fill counts"
    assert off.counts == {}, f"{label}: diagnostics=False must give counts={{}}"


# ---------------------------------------------------------------------------
# feature equivalence (chem / spatial / angle / combined)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(_batches()))
def test_chem_features_identical_when_diagnostics_false(name: str) -> None:
    batch = _batches()[name]
    _assert_identical(build_chem_graph(batch),
                      build_chem_graph(batch, diagnostics=False),
                      f"chem/{name}")


@pytest.mark.parametrize("name", sorted(_batches()))
def test_spatial_features_identical_when_diagnostics_false(name: str) -> None:
    batch = _batches()[name]
    on = build_spatial_graph(batch)
    off = build_spatial_graph(batch, diagnostics=False)
    _assert_identical(on, off, f"spatial/{name}")
    # chunked path stays consistent under the flag as well
    off_chunk = build_spatial_graph(batch, chunk_size=3, diagnostics=False)
    _assert_identical(on, off_chunk, f"spatial-chunk/{name}")


@pytest.mark.parametrize("name", sorted(_batches()))
def test_angle_features_identical_when_diagnostics_false(name: str) -> None:
    batch = _batches()[name]
    on = build_angle_triplets(batch)
    off = build_angle_triplets(batch, diagnostics=False)
    _assert_identical(on, off, f"angle/{name}")
    off_chunk = build_angle_triplets(batch, chunk_size=3, diagnostics=False)
    _assert_identical(on, off_chunk, f"angle-chunk/{name}")


@pytest.mark.parametrize("name", ["sn2", "mixed", "dense", "stretched"])
def test_legal_views_identical_when_diagnostics_false(name: str) -> None:
    batch = _batches()[name]
    on = build_legal_neighbor_views(batch)
    off = build_legal_neighbor_views(batch, diagnostics=False)
    _assert_identical(on.chem, off.chem, f"views-chem/{name}")
    _assert_identical(on.spatial, off.spatial, f"views-spatial/{name}")


def test_mixed_batch_diagnostics_false_all_views() -> None:
    """Multi-graph mixed batch: all three views identical, counts empty."""
    batch = sn2_bidirectional_batch(n_graphs=2)
    on_v = build_legal_neighbor_views(batch)
    off_v = build_legal_neighbor_views(batch, diagnostics=False)
    on_t = build_angle_triplets(batch)
    off_t = build_angle_triplets(batch, diagnostics=False)
    _assert_identical(on_v.chem, off_v.chem, "mixed/chem")
    _assert_identical(on_v.spatial, off_v.spatial, "mixed/spatial")
    _assert_identical(on_t, off_t, "mixed/angle")
    # nothing bridges graphs in the cheap path either
    graph_index = batch["graph_index"]
    for edge in (off_v.chem.edge_index, off_v.spatial.edge_index):
        assert bool((graph_index[edge[0]] == graph_index[edge[1]]).all())
    assert bool((graph_index[off_t.center_index].unsqueeze(1)
                 == graph_index[off_t.side_index]).all())


def test_internal_view_builders_honor_diagnostics() -> None:
    """``read_forward_batch`` + ``build_*_view`` (internal API) honour
    the flag exactly like the public wrappers; ``n_graphs`` (counts-only
    field) is 0 on the cheap read."""
    batch = sn2_forward_batch()
    data_on = read_forward_batch(batch, with_chem_scalars=True,
                                 diagnostics=True)
    data_off = read_forward_batch(batch, with_chem_scalars=True,
                                  diagnostics=False)
    assert data_on.n_graphs == 1
    assert data_off.n_graphs == 0        # counts-only field skipped
    # features of the read itself are identical
    assert torch.equal(data_on.pos, data_off.pos)
    assert torch.equal(data_on.edge_index, data_off.edge_index)

    chem_on = build_chem_view(data_on, diagnostics=True)
    chem_off = build_chem_view(data_off, diagnostics=False)
    _assert_identical(chem_on, chem_off, "internal/chem")
    spatial_on = build_spatial_view(data_on, diagnostics=True)
    spatial_off = build_spatial_view(data_off, diagnostics=False)
    _assert_identical(spatial_on, spatial_off, "internal/spatial")
    # internal == public on the same flag (both diagnostics=False)
    public_off = build_chem_graph(batch, diagnostics=False)
    _assert_tensors_identical(chem_off, public_off, "internal-vs-public/chem")
    public_spatial_off = build_spatial_graph(batch, diagnostics=False)
    _assert_tensors_identical(spatial_off, public_spatial_off,
                              "internal-vs-public/spatial")


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------
def test_gradients_identical_when_diagnostics_false() -> None:
    """Backward through both modes yields bitwise-equal, finite position
    gradients (the skipped count work never touches the autograd graph)."""
    base = sn2_bidirectional_batch(n_graphs=2)
    grads: list[torch.Tensor] = []
    for mode in (True, False):
        pos = base[NODE_POS_KEY].detach().clone().requires_grad_(True)
        batch = dict(base)
        batch[NODE_POS_KEY] = pos
        views = build_legal_neighbor_views(batch, diagnostics=mode)
        trip = build_angle_triplets(batch, diagnostics=mode)
        # outputs stay differentiable on the cheap path
        assert views.chem.rbf12.grad_fn is not None
        assert views.spatial.rbf12.grad_fn is not None
        assert trip.cos_basis.grad_fn is not None
        loss = (views.chem.rbf12.sum() + views.chem.distance.sum()
                + views.spatial.rbf12.sum() + views.spatial.distance.sum()
                + trip.cos_basis.sum() + trip.side_rbf12.sum()
                + trip.side_chem_attr.sum())
        loss.backward()
        assert pos.grad is not None
        assert torch.isfinite(pos.grad).all()
        grads.append(pos.grad.clone())
    assert torch.equal(grads[0], grads[1])


# ---------------------------------------------------------------------------
# count-only helpers are NOT called on the cheap path (monkeypatch proof)
# ---------------------------------------------------------------------------
def _boom(name: str):
    def _raise(*_args, **_kwargs):
        raise AssertionError(
            f"count-only helper {name} called with diagnostics=False")
    return _raise


def _patch_count_only_helpers(monkeypatch, replacement) -> None:
    """Patch every namespace the count-only helpers are bound in."""
    monkeypatch.setattr(provider_common, "count_graphs", replacement)
    monkeypatch.setattr(provider_common, "diagnostic_sum", replacement)
    monkeypatch.setattr(legal_module, "diagnostic_sum", replacement)
    monkeypatch.setattr(angle_module, "diagnostic_sum", replacement)


def test_count_only_helpers_not_called_when_diagnostics_false(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """diagnostics=False never routes through count_graphs /
    diagnostic_sum (the GPU→host sync helpers) on ANY public builder."""
    _patch_count_only_helpers(monkeypatch, _boom("count-only"))
    for batch in _batches().values():
        build_chem_graph(batch, diagnostics=False)
        build_spatial_graph(batch, diagnostics=False)
        build_legal_neighbor_views(batch, diagnostics=False)
        build_angle_triplets(batch, diagnostics=False)


def test_count_only_helpers_are_used_when_diagnostics_true(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: the default path DOES call the count-only helpers (so
    the proof above is not vacuous)."""
    calls: list[str] = []
    real_sum = provider_common.diagnostic_sum
    real_graphs = provider_common.count_graphs

    def spy_sum(values):
        calls.append("diagnostic_sum")
        return real_sum(values)

    def spy_graphs(graph_index):
        calls.append("count_graphs")
        return real_graphs(graph_index)

    _patch_count_only_helpers(monkeypatch, spy_sum)
    monkeypatch.setattr(provider_common, "count_graphs", spy_graphs)

    batches = _batches()
    build_chem_graph(batches["sn2"])                    # count_graphs + sum
    assert "count_graphs" in calls and "diagnostic_sum" in calls
    calls.clear()
    build_spatial_graph(batches["stretched"])           # beyond-cutoff sum
    assert "diagnostic_sum" in calls
    calls.clear()
    build_angle_triplets(batches["shell_tie"])          # boundary + cos sums
    assert calls.count("diagnostic_sum") >= 1
    assert "count_graphs" in calls
    # and the counts they produced are still the frozen ones
    counts = build_spatial_graph(batches["stretched"]).counts
    assert counts["n_input_pairs_beyond_cutoff"] == 1.0


# ---------------------------------------------------------------------------
# legality checks are NOT skipped when diagnostics=False
# ---------------------------------------------------------------------------
def _cross_graph_batch() -> dict:
    batch = sn2_bidirectional_batch(n_graphs=2)
    edge = batch[EDGE_INDEX_KEY].clone()
    n_total = int(batch[NODE_POS_KEY].shape[0])
    edge[1, 0] = (int(edge[0, 0]) + 6) % n_total
    out = dict(batch)
    out[EDGE_INDEX_KEY] = edge
    return out


def _mismatch_batch() -> dict:
    base = sn2_forward_batch()
    out = dict(base)
    out[EDGE_INDEX_KEY] = torch.cat(
        [base[EDGE_INDEX_KEY], base[EDGE_INDEX_KEY][:, [1]]], dim=1)
    for key in (EDGE_ATTR_KEY, EDGE_B_SRC_KEY, EDGE_B_TGT_KEY,
                EDGE_DELTA_B_KEY, EDGE_ABS_DELTA_KEY, EDGE_IS_CHANGED_KEY):
        value = base[key][[1]]
        if key == EDGE_DELTA_B_KEY:
            value = -value
        out[key] = torch.cat([base[key], value], dim=0)
    return out


def test_legality_checks_still_run_when_diagnostics_false() -> None:
    from tests.pesn.test_v541_feature_providers_fixtures import (
        zero_length_batch,
    )

    # zero-length angle side — explicit requirement of this fix
    with pytest.raises(FeatureProviderError) as excinfo:
        build_angle_triplets(zero_length_batch(), diagnostics=False)
    assert excinfo.value.code == "V541_PROVIDER_ZERO_DISTANCE"

    cross = _cross_graph_batch()
    for builder in (build_chem_graph, build_spatial_graph,
                    build_angle_triplets, build_legal_neighbor_views):
        with pytest.raises(FeatureProviderError) as excinfo:
            builder(cross, diagnostics=False)
        assert excinfo.value.code == "V541_PROVIDER_CROSS_GRAPH"

    with pytest.raises(FeatureProviderError) as excinfo:
        build_chem_graph(_mismatch_batch(), diagnostics=False)
    assert excinfo.value.code == "V541_PROVIDER_EDGE_ATTR_MISMATCH"

    nonfinite = dict(sn2_forward_batch())
    pos = nonfinite[NODE_POS_KEY].clone()
    pos[0, 0] = float("nan")
    nonfinite[NODE_POS_KEY] = pos
    for builder in (build_chem_graph, build_spatial_graph,
                    build_angle_triplets):
        with pytest.raises(FeatureProviderError) as excinfo:
            builder(nonfinite, diagnostics=False)
        assert excinfo.value.code == "V541_PROVIDER_GEOM_NONFINITE"


def test_default_flag_is_diagnostics_true() -> None:
    """The flag defaults to True — existing callers/tests unchanged."""
    batch = sn2_forward_batch()
    assert build_chem_graph(batch).counts
    assert build_spatial_graph(batch).counts
    assert build_angle_triplets(batch).counts
    assert build_legal_neighbor_views(batch).chem.counts
