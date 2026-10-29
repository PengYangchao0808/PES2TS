"""WP5-CERT streaming loader certificate (V5.4.1 §8.1 补证优先规则).

CPU-only, tiny synthetic fixtures — no real 50k data, no GPU:

* :class:`StreamThinViewV3` — compact columns / probe records / geometry
  are bitwise-equal to the full v1 loader on the mini fixture;
* :class:`NpzColumnMemmaps` — every member equals ``np.load``'s view;
* ``_pool_probe_rows`` — first/last/random all filter the pool FIRST; a
  dev/confirm position can never enter any probe;
* ``run_certificate_streaming`` — passes on the fixture stack (with a
  50k-tier split manifest), writes the v541-audits report path, binds the
  preregistration fail-closed, and its sampler/feature results equal the
  full-mode certificate's on the same fixture;
* full-mode default — ``run_certificate`` / ``main`` defaults are
  byte-identical to the pre-streaming behaviour (old report path/schema).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    ShardedKeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (  # noqa: E402
    GEOM_DIRNAME,
)
from pes2ts_core.pesn.offline.data.key_atom_stream_view import (  # noqa: E402
    CompactCurveDirView,
    NpzColumnMemmaps,
    StreamThinViewError,
    StreamThinViewV3,
)
from scripts.agent_ops.data_gate import (  # noqa: E402
    r3_v53mp_loader_v2_certificate as l2c,
)
from scripts.agent_ops.data_gate.r3_v53mp_build_geometry_pack import (  # noqa: E402
    build_geometry_pack,
)
from tests.pesn.test_r3_v53mp_dataset import (  # noqa: E402
    ALL_STATUSES,
    CURVE_IDS,
    _write_fixture,
)

SPLIT_NAME = "split_manifest_v53mp.json"


def _write_split_manifest(tables: Path, *, pool_50k: list[str],
                          pool_20k: list[str] | None = None,
                          dev: list[str] | None = None,
                          confirm: list[str] | None = None) -> dict:
    doc = {
        "schema": "v53mp_split_v1",
        "subsets": {
            "pool": {"20k": pool_20k or pool_50k, "50k": pool_50k},
            "dev": dev or [],
            "confirm": confirm or [],
        },
        "curve_folds": {cid: 0 for cid in pool_50k},
        "frozen": True,
    }
    (tables / SPLIT_NAME).write_text(
        json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return doc


def _build_stream_stack(tables: Path, split_doc: dict) -> dict:
    """shards → label cache (bound to the REAL manifest sha) → idx_v2 →
    geom pack — the streaming cert's fail-closed hash chain demands the
    cache binding be the live manifest sha (as in the real builds)."""
    import importlib.util

    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import _sha256
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        write_geometry_shards,
    )
    from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (
        build_endpoint_labels_spec_v1,
    )
    from pes2ts_core.pesn.offline.data.key_atom_label_cache_v53mp import (
        save_label_cache,
    )

    write_geometry_shards(tables)
    spec = importlib.util.spec_from_file_location(
        "i1_builder_stream",
        REPO / "scripts/agent_ops/data_gate/r3_v53mp_build_idx_v2.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    v1 = ShardedKeyAtomDatasetV3(
        tables, label_statuses=ALL_STATUSES, verify_manifest=True,
        light_cache=True)
    labels = v1.v52_labels()
    endpoint = build_endpoint_labels_spec_v1(v1, labels)
    manifest_sha = _sha256(tables / "data_manifest.json")
    save_label_cache(tables, labels, endpoint,
                     dataset_manifest_sha256=manifest_sha)
    builder.build_idx_v2(tables, tables / "idx_v2", verify_sources=False)
    build_geometry_pack(tables, tables / GEOM_DIRNAME)
    return {"v1": v1, "split_doc": split_doc}


@pytest.fixture()
def stream_stack(tmp_path: Path) -> dict:
    tables = _write_fixture(tmp_path / "data")
    split_doc = _write_split_manifest(
        tables, pool_50k=list(CURVE_IDS), pool_20k=list(CURVE_IDS),
        dev=[], confirm=["RXN_confirm_never_built"])
    out = _build_stream_stack(tables, split_doc)
    out["tables"] = tables
    return out


# ---------------------------------------------------------------------------
# StreamThinViewV3 vs the full v1 loader (bitwise)
# ---------------------------------------------------------------------------
def test_stream_columns_and_records_bitwise_equal_v1(stream_stack) -> None:
    tables = stream_stack["tables"]
    v1 = stream_stack["v1"]
    view = StreamThinViewV3(tables, label_statuses=ALL_STATUSES)
    cols = view.stream_columns()
    assert cols.n_rows == len(v1.samples)
    assert cols.weight_curves_checked == len(v1._curves)

    compact = view.compact_view(cols)
    assert isinstance(compact, CompactCurveDirView)
    assert len(compact) == len(v1.samples)
    for i, rec in enumerate(v1.samples):
        c = compact[i]
        assert (c.curve_id, c.fold, c.intent.direction) == \
               (rec.curve_id, rec.fold, rec.intent.direction)

    probes = np.asarray([0, 1, 5, len(v1.samples) - 1], dtype=np.int64)
    recs = view.probe_records(probes, cols)
    assert sorted(recs) == [int(p) for p in probes]
    for p in probes:
        a, b = recs[int(p)], v1.samples[int(p)]
        assert a.sample_id == b.sample_id
        assert a.curve_id == b.curve_id and a.node_index == b.node_index
        assert a.intent == b.intent
        assert a.atomic_numbers == b.atomic_numbers
        assert a.target_action_u == b.target_action_u
        assert a.fold == b.fold and a.label_status == b.label_status
        assert (a.label_weight, a.curve_weight, a.sample_weight) == \
               (b.label_weight, b.curve_weight, b.sample_weight)


def test_stream_geometry_lru_equal_v1(stream_stack) -> None:
    tables = stream_stack["tables"]
    v1 = stream_stack["v1"]
    view = StreamThinViewV3(tables, label_statuses=ALL_STATUSES,
                            cache_curves=1)
    cols = view.stream_columns()
    probes = np.asarray([0, len(v1.samples) - 1], dtype=np.int64)
    recs = view.probe_records(probes, cols)
    for p in probes:
        s1 = view.to_sample_v3(recs[int(p)])
        s2 = v1.to_sample_v3(v1.samples[int(p)])
        assert s1.sample_id == s2.sample_id
        assert np.array_equal(np.asarray(s1.xyz), np.asarray(s2.xyz))
        assert s1.atomic_numbers == s2.atomic_numbers
    # LRU bound respected (cache_curves=1)
    assert view.frames_lru_size() <= 1


def test_stream_view_typed_refusals(stream_stack) -> None:
    tables = stream_stack["tables"]
    view = StreamThinViewV3(tables, label_statuses=ALL_STATUSES)
    with pytest.raises(StreamThinViewError):
        view.probe_records(np.asarray([10_000_000], dtype=np.int64),
                           view.stream_columns())


# ---------------------------------------------------------------------------
# NpzColumnMemmaps vs np.load
# ---------------------------------------------------------------------------
def test_npz_memmap_members_equal_numpy_load(stream_stack) -> None:
    tables = stream_stack["tables"]
    npz_path = tables / "labels_cache_v53mp.npz"
    mm = NpzColumnMemmaps(npz_path)
    ref = np.load(npz_path, allow_pickle=False)
    assert set(mm.names) == set(ref.files)
    for name in ref.files:
        assert np.array_equal(np.asarray(mm.column(name)),
                              np.asarray(ref[name])), name


def test_npz_memmap_missing_member_typed(stream_stack) -> None:
    mm = NpzColumnMemmaps(stream_stack["tables"] / "labels_cache_v53mp.npz")
    with pytest.raises(StreamThinViewError):
        mm.column("no_such_column")
    with pytest.raises(StreamThinViewError):
        NpzColumnMemmaps(stream_stack["tables"] / "missing.npz")


def test_sid_rows_handles_unsorted_cache(tmp_path: Path) -> None:
    """Regression: the delivered 50k label cache is NOT sorted by sid
    (curve/node numeric order, `:9:` before `:10:`), so the sid→row map
    must be order-independent — never searchsorted."""
    rng = np.random.RandomState(7)
    sids = [f"RXN_{i:010d}:0:RXN_{i:010d}_fwd:0.08" for i in range(500)]
    order = rng.permutation(len(sids))          # deliberately unsorted
    shuffled = [sids[i] for i in order]
    ids = np.asarray(shuffled, dtype="U64")
    probes = [sids[3], sids[250], sids[499], sids[0]]
    rows = l2c._sid_rows(ids, probes)
    got = [shuffled[int(r)] for r in rows]
    assert got == probes                          # aligned to probe order
    # sorted column works too
    ids_sorted = np.asarray(sorted(sids), dtype="U64")
    rows2 = l2c._sid_rows(ids_sorted, probes)
    assert [sorted(sids)[int(r)] for r in rows2] == probes
    # a missing sid is a typed refusal
    with pytest.raises(l2c.StreamCertError) as exc:
        l2c._sid_rows(ids, ["RXN_nope:0:x:0.08"])
    assert exc.value.code == "EVIDENCE_MISSING"


# ---------------------------------------------------------------------------
# pool-only probe selection
# ---------------------------------------------------------------------------
def test_pool_probe_rows_filters_pool_first() -> None:
    # dataset positions 0..9; pool = {2, 4, 6}; dev = {1, 9}
    pool = np.asarray([2, 4, 6], dtype=np.int64)
    rng = np.random.RandomState(17)
    probes = l2c._pool_probe_rows(pool, rng)
    assert set(int(p) for p in probes) <= {2, 4, 6}
    # first/last of the POOL (not of the dataset) are always included
    assert 2 in probes and 6 in probes
    # a huge pool: probes are a strict subset and always pool positions
    big_pool = np.arange(1000, 5000, dtype=np.int64)
    probes2 = l2c._pool_probe_rows(big_pool, np.random.RandomState(17))
    assert set(int(p) for p in probes2) <= set(int(p) for p in big_pool)
    assert 1000 in probes2 and 4999 in probes2
    # ~1% random + 2*1000 edges
    assert len(probes2) >= 2000 + 40


# ---------------------------------------------------------------------------
# streaming certificate on the fixture
# ---------------------------------------------------------------------------
def _write_prereg(path: Path, report_path: Path) -> dict:
    doc = {
        "schema": "v541_wp5_cert_preregistration_v1",
        "mode": "streaming_phased",
        "tier": "50k",
        "attempt": 1,
        "probe_protocol": {
            "seed": 17,
            "pool_only": True,
            "sampler_updates": 200,
            "feature_probes": 1000,
        },
        "cgroup": {"memory_max": "12G", "memory_high": "10G", "swap_max": 0},
        "outputs": {"certificate": str(report_path)},
    }
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def test_streaming_certificate_passes_on_fixture(stream_stack,
                                                 tmp_path: Path) -> None:
    tables = stream_stack["tables"]
    report = tmp_path / "loader_v2_streaming_certificate.json"
    prereg = tmp_path / "wp5_cert_preregistration.json"
    _write_prereg(prereg, report)
    verdict = l2c.run_certificate_streaming(
        tables, report, tier="50k", prereg_path=prereg)
    assert verdict["passed"] is True, json.dumps(verdict, indent=1)[:4000]
    assert verdict["schema"] == "v53mp_loader_v2_streaming_certificate_v1"
    assert verdict["mode"] == "streaming"
    assert verdict["sampler_id"] == "v53mp_curve_dir_v1"
    assert verdict["failure"] is None
    assert report.is_file()
    assert verdict["preregistration_sha256"] == hashlib.sha256(
        prereg.read_bytes()).hexdigest()
    # every check passed
    for name, chk in verdict["checks"].items():
        assert chk["passed"] is True, name
    # phase accounting present (columns / sampler / features)
    names = [p["phase"] for p in verdict["phases"]]
    assert {"v2_open", "hash_verify", "columns", "sampler",
            "features"} <= set(names)
    # pool-only probe contract: confirm id never probed, pool curves only
    contract = verdict["pool_probe_contract"]
    assert contract["pool_curves"] == len(CURVE_IDS)
    assert contract["pool_only"] is True
    assert contract["feature_probes"] > 0
    # hash chain verified
    hv = verdict["hash_verification"]
    assert hv["manifest_sha_matches_idx_binding"] is True
    assert hv["label_cache_binding_matches_manifest"] is True
    assert hv["split_matches_idx_binding"] is True
    for meta in hv["files"].values():
        assert meta["match"] is True


def test_streaming_and_full_agree_on_fixture(stream_stack,
                                             tmp_path: Path) -> None:
    """The streaming certificate's sampler/feature outcomes equal the full
    mode's on the same fixture (same seeds, same pool, same positions)."""
    tables = stream_stack["tables"]
    report_s = tmp_path / "stream.json"
    prereg = tmp_path / "prereg.json"
    _write_prereg(prereg, report_s)
    vs = l2c.run_certificate_streaming(
        tables, report_s, tier="50k", prereg_path=prereg)
    report_dir = tmp_path / "full_reports"
    vf = l2c.run_certificate(tables, report_dir, tier="50k")
    assert vs["passed"] is True and vf["passed"] is True
    # sampler: identical split sizes + zero mismatches on both paths
    assert vs["checks"]["sampler_equality"]["train_positions"] == \
           vf["checks"]["sampler_equality"]["train_positions"]
    assert vs["checks"]["sampler_equality"]["inner_val_positions"] == \
           vf["checks"]["sampler_equality"]["inner_val_positions"]
    assert vs["checks"]["sampler_equality"]["position_mismatches"] == 0
    assert vf["checks"]["sampler_equality"]["position_mismatches"] == 0
    # features: identical counts and levels
    assert vs["checks"]["feature_equality"]["feature_level"] == \
           vf["checks"]["feature_equality"]["feature_level"]
    assert vs["checks"]["feature_equality"]["positions"] == \
           min(1000, vs["pool_probe_contract"]["pool_positions"])
    # columns: both checked every surviving row's equality at their probes
    assert vs["checks"]["column_equality"]["count_equal"] is True
    assert vf["checks"]["column_equality"]["count_equal"] is True


def test_streaming_certificate_prereg_mismatch_refused(stream_stack,
                                                        tmp_path: Path) -> None:
    tables = stream_stack["tables"]
    report = tmp_path / "cert.json"
    prereg = tmp_path / "prereg.json"
    doc = _write_prereg(prereg, report)
    doc["mode"] = "full"                      # tampered after freeze
    prereg.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    verdict = l2c.run_certificate_streaming(
        tables, report, tier="50k", prereg_path=prereg)
    assert verdict["passed"] is False
    assert verdict["failure"]["code"] == "PREREG_MISMATCH"
    # the report is still written (evidence even on refusal)
    assert report.is_file()


def test_streaming_certificate_missing_prereg_refused(stream_stack,
                                                      tmp_path: Path) -> None:
    tables = stream_stack["tables"]
    verdict = l2c.run_certificate_streaming(
        tables, tmp_path / "cert.json", tier="50k",
        prereg_path=tmp_path / "nope.json")
    assert verdict["passed"] is False
    assert verdict["failure"]["code"] == "PREREG_MISMATCH"


def test_streaming_failure_preserves_partial_checks(stream_stack,
                                                    tmp_path: Path,
                                                    monkeypatch) -> None:
    """A mid-run typed failure keeps the already-completed checks in the
    report (the attempt-1 real run lost its passing columns evidence)."""
    tables = stream_stack["tables"]
    report = tmp_path / "cert.json"
    prereg = tmp_path / "prereg.json"
    _write_prereg(prereg, report)

    def boom(*args, **kwargs):
        raise l2c.StreamCertError("EVIDENCE_MISSING", "injected for test")

    monkeypatch.setattr(l2c, "_stream_check_labels", boom)
    verdict = l2c.run_certificate_streaming(
        tables, report, tier="50k", prereg_path=prereg)
    assert verdict["passed"] is False
    assert verdict["failure"]["code"] == "EVIDENCE_MISSING"
    # the columns check completed before the injected failure — preserved
    assert "checks" in verdict
    assert verdict["checks"]["column_equality"]["passed"] is True
    # report on disk matches the in-memory verdict
    on_disk = json.loads(report.read_text(encoding="utf-8"))
    assert on_disk["checks"]["column_equality"]["passed"] is True


# ---------------------------------------------------------------------------
# full-mode default unchanged
# ---------------------------------------------------------------------------
def test_full_mode_default_unchanged(stream_stack,
                                     tmp_path: Path) -> None:
    tables = stream_stack["tables"]
    report_dir = tmp_path / "reports"
    verdict = l2c.run_certificate(tables, report_dir, tier="50k")
    assert verdict["schema"] == "v53mp_loader_v2_certificate_v1"
    assert "mode" not in verdict          # full mode report shape unchanged
    # old report naming: loader_v2_certificate_<sha12>.json
    files = sorted(p.name for p in report_dir.iterdir())
    assert len(files) == 1 and \
        files[0].startswith("loader_v2_certificate_") and \
        files[0].endswith(".json")
    # constants unchanged
    assert l2c.DEFAULT_TIER == "20k"
    assert l2c.PROTOCOL["seed"] == 17
    assert l2c.PROTOCOL["sampler_updates"] == 200
    assert l2c.PROTOCOL["feature_probes"] == 1000


def test_main_default_mode_is_full(tmp_path: Path, capsys) -> None:
    """``--mode`` defaults to full; ``--tier 60k`` still exits 2 first."""
    with pytest.raises(SystemExit) as exc:
        l2c.main(["--tier", "60k"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc2:
        l2c.main(["--mode", "bogus"])
    assert exc2.value.code == 2
