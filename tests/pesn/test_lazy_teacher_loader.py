"""Lazy/sharded teacher loader: equivalence, batching, isolation, cache bounds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.pesn.conftest import build_synthetic_reaction_data  # noqa: E402
from pes2ts_core.pesn.offline.data.dataset import (  # noqa: E402
    ReactionRecord,
    build_dataset_manifest,
    task_coverage_of,
    write_shards,
)
from pes2ts_core.pesn.offline.labels.direction import (  # noqa: E402
    compute_direction_labels,
    evaluate_distance_coords,
)
from pes2ts_core.pesn.offline.labels.progress import (  # noqa: E402
    compute_progress_labels,
    masses_for,
)
from pes2ts_core.pesn.offline.training import loader as loader_module  # noqa: E402
from pes2ts_core.pesn.offline.training.loader import (  # noqa: E402
    LazyReactionSequence,
    LazyTeacherProvider,
    LoaderError,
    ReactionData,
    coordinate_spec_from_doc,
    epoch_batch_indices,
    epoch_batches,
    load_teacher_reactions,
    make_batch,
    task_graph_from_doc,
)
from pes2ts_core.utils.jsonio import write_json  # noqa: E402


def _graph_doc(task_graph) -> dict:
    return {
        "atomic_numbers": list(task_graph.atomic_numbers),
        "atom_map": list(task_graph.atom_map),
        "reactant_bonds": [[t.i, t.j, t.order] for t in task_graph.reactant_bonds],
        "product_bonds": [[t.i, t.j, t.order] for t in task_graph.product_bonds],
        "edits": [
            [e.kind, e.i, e.j, e.reactant_order, e.product_order]
            for e in task_graph.edits
        ],
        "source": task_graph.source,
        "provenance": dict(task_graph.provenance),
    }


def _coord_doc(spec) -> dict:
    return {
        "spec_version": spec.spec_version,
        "coords": [
            {
                "kind": c.kind,
                "atoms": list(c.atoms),
                "units": c.units,
                "role": c.role,
                "scale_a": c.scale_a,
                "ref_q": c.ref_q,
                "label": c.label,
            }
            for c in spec.coords
        ],
    }


def _record_for(reaction_id: str) -> ReactionRecord:
    data = build_synthetic_reaction_data(reaction_id)
    task_graph = data.task_graph
    masses = masses_for(task_graph.atomic_numbers)
    progress = compute_progress_labels(
        data.coords,
        masses,
        ts_frame_index=0,
        splice_index=data.coords.shape[0] // 2,
        layout="ts_first_dual_branch",
        orientation="P_first",
    )
    q = evaluate_distance_coords(data.coords, [(0, 1), (2, 3)])
    direction = compute_direction_labels(q, progress)
    return ReactionRecord(
        reaction_id=reaction_id,
        coords=data.coords.astype(np.float32),
        atomic_numbers=np.asarray(task_graph.atomic_numbers, dtype=int),
        task_graph_doc=_graph_doc(task_graph),
        coordinate_spec_doc=_coord_doc(data.coordinate_spec),
        progress=progress,
        direction=direction,
        provenance={"source_method": "synthetic", "source_ref": "tests/pesn"},
        exclusion=None,
    )


def _excluded_record(reaction_id: str) -> ReactionRecord:
    return ReactionRecord(
        reaction_id=reaction_id,
        coords=np.zeros(0, dtype=np.float32),
        atomic_numbers=np.zeros(0, dtype=int),
        task_graph_doc={},
        coordinate_spec_doc={},
        progress=None,
        direction=None,
        provenance={"reason": "fixture_exclusion"},
        exclusion="FIXTURE_EXCLUDED",
    )


def _write_dataset(
    dataset_dir: Path,
    *,
    fit_ids: list[str],
    dev_ids: list[str],
    excluded_ids: list[str] = (),
    shard_size: int = 2,
    extra_subset_members: dict[str, list[str]] | None = None,
) -> None:
    records = [_record_for(rid) for rid in fit_ids]
    records += [_record_for(rid) for rid in dev_ids]
    records += [_excluded_record(rid) for rid in excluded_ids]
    shards = write_shards(records, dataset_dir, shard_size=shard_size)
    manifest = build_dataset_manifest(
        dataset_dir=dataset_dir,
        shards=shards,
        source_hashes={"fixture": "0" * 64},
        parent_split_sha256="1" * 64,
        label_spec_sha256="2" * 64,
        n_reactions=len(records),
        n_excluded=len(excluded_ids),
        task_coverage=task_coverage_of(records),
    )
    subset_members = {
        "train_fit": sorted(fit_ids),
        "train_dev": sorted(dev_ids),
    }
    if extra_subset_members:
        subset_members.update(extra_subset_members)
    manifest["subset_members"] = subset_members
    write_json(dataset_dir / "dataset_manifest.json", manifest)


def _legacy_load(dataset_dir: Path) -> list[ReactionData]:
    """Pre-change eager loader, kept as an independent oracle for equivalence."""
    root = Path(dataset_dir)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    reactions: list[ReactionData] = []
    for shard in manifest.get("shards", []):
        tag = str(shard["shard"])
        states = np.load(root / f"states_{tag}.npz")
        labels = np.load(root / f"labels_{tag}.npz")
        provenance = json.loads(
            (root / f"provenance_{tag}.json").read_text(encoding="utf-8")
        )
        for rid, prov in provenance["reactions"].items():
            if prov.get("exclusion") is not None:
                continue
            reactions.append(
                ReactionData(
                    reaction_id=rid,
                    atomic_numbers=states[f"atomic_numbers__{rid}"].astype(np.int64),
                    coords=states[f"coords__{rid}"].astype(np.float64),
                    task_graph=task_graph_from_doc(prov["task_graph"]),
                    coordinate_spec=coordinate_spec_from_doc(prov["coordinate_spec"]),
                    s=labels[f"s__{rid}"].astype(np.float64),
                    d=labels[f"d__{rid}"].astype(np.float64),
                    s_mask=labels[f"s_mask__{rid}"].astype(bool),
                    side=labels[f"side__{rid}"].astype(np.int64),
                    d_arc=labels[f"d_arc__{rid}"].astype(np.float64),
                    direction=labels[f"direction__{rid}"].astype(np.float64),
                    activity=labels[f"activity__{rid}"].astype(np.float64),
                    dir_mask=labels[f"dir_mask__{rid}"].astype(bool),
                    act_mask=labels[f"act_mask__{rid}"].astype(bool),
                    method=str(prov.get("source_method", "unknown")),
                )
            )
    return reactions


def _assert_reaction_equal(left: ReactionData, right: ReactionData) -> None:
    assert left.reaction_id == right.reaction_id
    assert left.method == right.method
    np.testing.assert_array_equal(left.atomic_numbers, right.atomic_numbers)
    np.testing.assert_array_equal(left.coords, right.coords)
    np.testing.assert_array_equal(left.s, right.s)
    np.testing.assert_array_equal(left.d, right.d)
    np.testing.assert_array_equal(left.s_mask, right.s_mask)
    np.testing.assert_array_equal(left.side, right.side)
    np.testing.assert_array_equal(left.d_arc, right.d_arc)
    np.testing.assert_array_equal(left.direction, right.direction)
    np.testing.assert_array_equal(left.activity, right.activity)
    np.testing.assert_array_equal(left.dir_mask, right.dir_mask)
    np.testing.assert_array_equal(left.act_mask, right.act_mask)
    assert left.task_graph.atomic_numbers == right.task_graph.atomic_numbers
    assert left.task_graph.atom_map == right.task_graph.atom_map
    assert left.task_graph.edits == right.task_graph.edits
    assert left.coordinate_spec.spec_version == right.coordinate_spec.spec_version
    assert [c.label for c in left.coordinate_spec.coords] == [
        c.label for c in right.coordinate_spec.coords
    ]


FIT_IDS = [f"RXN_FIT{i:08d}" for i in (3, 1, 4, 0, 2)]
DEV_IDS = [f"RXN_DEV{i:08d}" for i in range(2)]


def test_lazy_provider_matches_legacy_full_load(tmp_path) -> None:
    excluded = ["RXN_EXCL00000001"]
    dataset = tmp_path / "ds"
    _write_dataset(
        dataset, fit_ids=FIT_IDS, dev_ids=DEV_IDS, excluded_ids=excluded, shard_size=2
    )
    legacy = _legacy_load(dataset)
    wrapper = load_teacher_reactions(dataset)
    provider = LazyTeacherProvider(dataset)
    lazy_list = list(provider.iter_reactions())
    canonical_ids = [r.reaction_id for r in legacy]
    assert [r.reaction_id for r in wrapper] == canonical_ids
    assert [r.reaction_id for r in lazy_list] == canonical_ids
    assert set(canonical_ids) == set(FIT_IDS) | set(DEV_IDS)
    assert all(r.reaction_id not in excluded for r in lazy_list)
    for oracle, wrapped, lazy in zip(legacy, wrapper, lazy_list):
        _assert_reaction_equal(oracle, wrapped)
        _assert_reaction_equal(oracle, lazy)
    legacy_fit = [r for r in legacy if r.reaction_id in set(FIT_IDS)]
    lazy_fit = list(provider.iter_reactions("train_fit"))
    assert [r.reaction_id for r in lazy_fit] == [r.reaction_id for r in legacy_fit]
    for oracle, lazy in zip(legacy_fit, lazy_fit):
        _assert_reaction_equal(oracle, lazy)


def test_epoch_batching_identical_eager_vs_lazy(tmp_path) -> None:
    dataset = tmp_path / "ds"
    _write_dataset(dataset, fit_ids=FIT_IDS, dev_ids=DEV_IDS, shard_size=2)
    provider = LazyTeacherProvider(dataset)
    eager_train = [
        r
        for r in _legacy_load(dataset)
        if r.reaction_id in set(FIT_IDS) and r.s_mask.any()
    ]
    lazy_train = LazyReactionSequence(
        provider, provider.valid_label_reaction_ids(subset="train_fit")
    )
    assert [r.reaction_id for r in eager_train] == lazy_train.reaction_ids

    def trajectory(chunks):
        rng = np.random.default_rng(42)
        out = []
        for chunk in chunks:
            batch = make_batch(
                chunk,
                frames_per_reaction=8,
                pairs_per_reaction=4,
                pair_margin=0.05,
                rng=rng,
            )
            out.append(
                {
                    "samples": [
                        (s.reaction_id, s.frame_index, s.s, s.d, s.valid)
                        for s in batch.samples
                    ],
                    "pairs": list(batch.pairs),
                }
            )
        return out

    eager_chunks: list = []
    lazy_chunks: list = []
    eager_orders: list = []
    lazy_orders: list = []
    for epoch in range(3):
        e_chunks = epoch_batches(
            eager_train,
            batch_reactions=2,
            frames_per_reaction=8,
            pairs_per_reaction=4,
            pair_margin=0.05,
            seed=42,
            epoch=epoch,
        )
        idx_chunks = epoch_batch_indices(
            len(lazy_train), batch_reactions=2, seed=42, epoch=epoch
        )
        l_chunks = [[lazy_train[int(i)] for i in idx] for idx in idx_chunks]
        eager_orders.append([[r.reaction_id for r in c] for c in e_chunks])
        lazy_orders.append([[r.reaction_id for r in c] for c in l_chunks])
        eager_chunks.extend(e_chunks)
        lazy_chunks.extend(l_chunks)
    assert eager_orders == lazy_orders
    assert trajectory(eager_chunks) == trajectory(lazy_chunks)


def test_subset_isolation_fit_never_touches_dev_shards(tmp_path, monkeypatch) -> None:
    fit = [f"RXN_FIT{i:08d}" for i in range(4)]
    dev = [f"RXN_DEV{i:08d}" for i in range(2)]
    dataset = tmp_path / "ds"
    _write_dataset(dataset, fit_ids=fit, dev_ids=dev, shard_size=2)
    opened: list[str] = []
    real_load = np.load

    def spy_load(file, *args, **kwargs):
        opened.append(Path(file).name)
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(loader_module.np, "load", spy_load)
    provider = LazyTeacherProvider(dataset)
    ids = provider.valid_label_reaction_ids(subset="train_fit")
    reactions = list(provider.iter_reactions("train_fit"))
    assert set(ids) == set(fit)
    assert [r.reaction_id for r in reactions] == provider.reaction_ids("train_fit")
    dev_shard_files = {"states_0002.npz", "labels_0002.npz"}
    assert opened, "fit iteration must open its own shards"
    assert dev_shard_files.isdisjoint(set(opened))


def test_shard_cache_bounds_iteration(tmp_path) -> None:
    fit = [f"RXN_FIT{i:08d}" for i in range(5)]
    dev = [f"RXN_DEV{i:08d}" for i in range(1)]
    dataset = tmp_path / "ds"
    _write_dataset(dataset, fit_ids=fit, dev_ids=dev, shard_size=1)
    for cache_size in (1, 2):
        provider = LazyTeacherProvider(dataset, shard_cache_size=cache_size)
        n_shards = len(provider.manifest["shards"])
        assert n_shards > cache_size
        for rid in provider.reaction_ids():
            provider.get_reaction(rid)
            assert provider.open_shard_count <= cache_size
        assert provider.open_shard_count <= cache_size


def test_extra_subsets_hard_fail_and_never_served(tmp_path) -> None:
    sealed_ids = ["RXN_SEALED000001"]
    dataset = tmp_path / "ds"
    _write_dataset(
        dataset,
        fit_ids=FIT_IDS[:2],
        dev_ids=DEV_IDS[:1],
        shard_size=2,
        extra_subset_members={"sealed": sealed_ids},
    )
    with pytest.raises(LoaderError):
        LazyTeacherProvider(dataset)
    provider = LazyTeacherProvider(dataset, allow_extra_subsets=True)
    assert provider.reaction_ids("train_fit") == sorted(FIT_IDS[:2])
    with pytest.raises(LoaderError):
        provider.reaction_ids("sealed")
    with pytest.raises(LoaderError):
        list(provider.iter_reactions("sealed"))


def test_config_accepts_train_lazy_loading(tmp_path) -> None:
    from pes2ts_core.pesn.offline.cli import OfflineConfigError, load_pesn_config

    good = tmp_path / "good.yaml"
    good.write_text(
        "schema_version: pesn_g3p1_config_v1\n"
        "train:\n"
        "  lazy_loading: true\n",
        encoding="utf-8",
    )
    loaded = load_pesn_config(good)
    assert loaded["train"]["lazy_loading"] is True
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "schema_version: pesn_g3p1_config_v1\n"
        "train:\n"
        "  lazy_loading_typo: true\n",
        encoding="utf-8",
    )
    with pytest.raises(OfflineConfigError):
        load_pesn_config(bad)
