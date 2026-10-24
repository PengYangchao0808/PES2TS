"""Dev feature pack tests (S4 plan §5.3 / lane I2).

Coverage:

* **blob codec roundtrip** — every value kind
  ``build_sample_features_v3`` produces (arrays of each dtype/shape,
  np.float32/64 scalars, strings, tuples, None) encodes+decodes
  byte-identically; corrupt/truncated blobs are typed refusals;
* **pack write/open roundtrip** — offsets integrity (monotone, last ==
  blob bytes), positions ascending, binding json fields, lazy view
  ``cache[position]`` equality vs a live build, ``expected_positions``
  coverage refusal, features-sha drift refusal;
* **trainer pack-vs-none equivalence** — a tiny synthetic run with
  ``dev_feature_pack=<pack>`` produces metrics EXACTLY equal to the same
  run without the pack (all floats bitwise), and the no-pack path stays
  byte-identical to pre-pack behaviour.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pes2ts_core.pesn.features.key_atom_graph_v3 import (  # noqa: E402
    build_sample_features_v3,
)
from pes2ts_core.pesn.offline.data.key_atom_dev_pack_v53mp import (  # noqa: E402
    DevFeaturePackView,
    DevPackError,
    blob_decode,
    blob_encode,
    features_sha256_of,
    positions_sha256,
    write_dev_feature_pack,
)


# ---------------------------------------------------------------------------
# blob codec
# ---------------------------------------------------------------------------
def _sample_dict() -> dict:
    """Every value kind the builder emits (post-finalize)."""
    return {
        "z": np.array([1, 6, 8], dtype=np.int64),
        "pos": np.arange(9, dtype=np.float32).reshape(3, 3),
        "edge_index": np.zeros((2, 4), dtype=np.int64),
        "pair_index": np.zeros((2, 0), dtype=np.int64),
        "y_teacher": np.full((2, 3), 0.25, dtype=np.float32),
        "frame_weight": np.float32(0.875),
        "d0": np.float64(0.08),
        "sample_id": "RXN_x:0:RXN_x_fwd:0.08",
        "curve_id": "RXN_x",
        "intent_id": "RXN_x_fwd",
        "label_status": "ok",
        "endpoint_names": ("A", "B"),
        "flag": True,
        "count": 3,
        "missing": None,
        "plain_float": 1.5,
    }


def test_blob_roundtrip_all_value_kinds() -> None:
    built = _sample_dict()
    blob = blob_encode(built)
    got = blob_decode(blob)
    assert set(got) == set(built)
    for key in built:
        va, vb = built[key], got[key]
        if isinstance(va, np.ndarray):
            assert isinstance(vb, np.ndarray), key
            assert va.dtype == vb.dtype, key
            assert va.shape == vb.shape, key
            assert va.tobytes() == vb.tobytes(), key
        elif isinstance(va, tuple):
            assert isinstance(vb, tuple) and va == vb, key
        elif isinstance(va, (np.float32, np.float64)):
            assert va.dtype == vb.dtype and va.tobytes() == vb.tobytes(), key
        else:
            assert va == vb and type(va) is type(vb), key


def test_blob_roundtrip_real_features(tmp_path: Path) -> None:
    """Real F2 features from the trainer's synthetic fixture roundtrip."""
    torch = pytest.importorskip("torch")
    del torch
    from tests.pesn.test_r3_v53mp_trainer import _build_env, _smoke_cfg
    env = _build_env(tmp_path, {"cv0": 1, "cv1": 1, "cv2": 1,
                                "cv_dev0": 0, "cv_dev1": 0})
    rec = env.ds.records([0])[0]
    built = build_sample_features_v3(
        env.ds.to_sample_v3(rec), level="F2", intent_off=False)
    got = blob_decode(blob_encode(built))
    assert set(got) == set(built)
    for key in built:
        va, vb = built[key], got[key]
        if isinstance(va, np.ndarray):
            assert va.dtype == vb.dtype and va.shape == vb.shape
            assert va.tobytes() == vb.tobytes(), key
        else:
            assert va == vb, key
    del _smoke_cfg


def test_blob_truncated_refused() -> None:
    blob = blob_encode(_sample_dict())
    with pytest.raises(DevPackError) as exc:
        blob_decode(blob[:10])
    assert exc.value.code == "BLOB_TRUNCATED"


def test_blob_body_length_refused() -> None:
    blob = blob_encode(_sample_dict())
    with pytest.raises(DevPackError) as exc:
        blob_decode(blob + b"\x00\x00")
    assert exc.value.code == "BLOB_BODY_LENGTH"


# ---------------------------------------------------------------------------
# pack write / open / lazy view
# ---------------------------------------------------------------------------
def _write_mini_pack(tmp_path: Path, positions: list[int],
                     blobs: list[bytes], **kwargs) -> Path:
    pack_dir = tmp_path / "pack"
    write_dev_feature_pack(
        pack_dir, positions, blobs,
        feature_level=kwargs.pop("feature_level", "F2"),
        intent_off=False,
        dataset_binding={"n_samples": 16}, **kwargs)
    return pack_dir


def test_pack_write_offsets_and_binding(tmp_path: Path) -> None:
    built = _sample_dict()
    blob = blob_encode(built)
    positions = [0, 3, 7]
    pack_dir = _write_mini_pack(tmp_path, positions, [blob] * 3)
    offsets = np.load(pack_dir / "offsets.npy", allow_pickle=False)
    pos = np.load(pack_dir / "positions.npy", allow_pickle=False)
    blobs = np.load(pack_dir / "blobs.npy", mmap_mode="r", allow_pickle=False)
    assert offsets.tolist() == [0, len(blob), 2 * len(blob), 3 * len(blob)]
    assert pos.tolist() == positions
    assert int(blobs.shape[0]) == 3 * len(blob)
    binding = json.loads(next(pack_dir.glob("*.json")).read_text("utf-8"))
    assert binding["schema"] == "v53mp_dev_pack_v1"
    assert binding["features_sha256"] == features_sha256_of()
    assert binding["positions_sha256"] == positions_sha256(positions)
    assert next(pack_dir.glob("*.json")).stem == binding["positions_sha256"]
    assert binding["n_positions"] == 3
    assert binding["feature_level"] == "F2"
    for meta in binding["files"].values():
        assert len(meta["sha256"]) == 64 and meta["bytes"] > 0


def test_pack_view_lazy_getitem(tmp_path: Path) -> None:
    built = _sample_dict()
    blob = blob_encode(built)
    positions = [2, 5]
    pack_dir = _write_mini_pack(tmp_path, positions, [blob, blob])
    view = DevFeaturePackView.open(pack_dir, feature_level="F2")
    assert len(view) == 2
    assert view.positions.tolist() == positions
    got = view[5]
    for key in built:
        va, vb = built[key], got[key]
        if isinstance(va, np.ndarray):
            assert va.tobytes() == vb.tobytes() and va.dtype == vb.dtype
        else:
            assert va == vb
    with pytest.raises(DevPackError) as exc:
        view[4]                       # not in the pack
    assert exc.value.code == "PACK_POSITION"


def test_pack_expected_positions_coverage(tmp_path: Path) -> None:
    blob = blob_encode(_sample_dict())
    pack_dir = _write_mini_pack(tmp_path, [0, 1], [blob, blob])
    # covered subset opens fine
    DevFeaturePackView.open(pack_dir, expected_positions=[0, 1])
    with pytest.raises(DevPackError) as exc:
        DevFeaturePackView.open(pack_dir, expected_positions=[0, 7])
    assert exc.value.code == "PACK_POSITIONS"


def test_pack_features_sha_drift_refused(tmp_path: Path) -> None:
    blob = blob_encode(_sample_dict())
    pack_dir = _write_mini_pack(tmp_path, [0], [blob])
    binding_path = next(pack_dir.glob("*.json"))
    binding = json.loads(binding_path.read_text("utf-8"))
    binding["features_sha256"] = "0" * 64
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(DevPackError) as exc:
        DevFeaturePackView.open(pack_dir)
    assert exc.value.code == "PACK_FEATURES_DRIFT"


def test_pack_level_mismatch_refused(tmp_path: Path) -> None:
    blob = blob_encode(_sample_dict())
    pack_dir = _write_mini_pack(tmp_path, [0], [blob])
    with pytest.raises(DevPackError) as exc:
        DevFeaturePackView.open(pack_dir, feature_level="F3")
    assert exc.value.code == "PACK_LEVEL"


def test_pack_file_sha_verification(tmp_path: Path) -> None:
    blob = blob_encode(_sample_dict())
    pack_dir = _write_mini_pack(tmp_path, [0, 1], [blob, blob])
    # tamper blobs.npy without updating the binding
    raw = bytearray((pack_dir / "blobs.npy").read_bytes())
    raw[-1] ^= 0xFF
    (pack_dir / "blobs.npy").write_bytes(bytes(raw))
    with pytest.raises(DevPackError) as exc:
        DevFeaturePackView.open(pack_dir, verify_files=True)
    assert exc.value.code == "PACK_SHA"


def test_pack_positions_order_refused(tmp_path: Path) -> None:
    blob = blob_encode(_sample_dict())
    with pytest.raises(DevPackError) as exc:
        _write_mini_pack(tmp_path, [5, 2], [blob, blob])
    assert exc.value.code == "PACK_POSITIONS_ORDER"


def test_pack_count_mismatch_refused(tmp_path: Path) -> None:
    blob = blob_encode(_sample_dict())
    with pytest.raises(DevPackError) as exc:
        _write_mini_pack(tmp_path, [0, 1], [blob])
    assert exc.value.code == "PACK_COUNT"


# ---------------------------------------------------------------------------
# binding discovery: migration ``*.sha256sidecar.json`` pollution
# ---------------------------------------------------------------------------
def _write_sidecars(pack_dir: Path) -> None:
    """Simulate the migration's sha256 cache sidecars next to pack files."""
    for name in ("blobs.npy", "offsets.npy", "positions.npy"):
        (pack_dir / (name + ".sha256sidecar.json")).write_text(
            json.dumps({"sha256": "0" * 64}), encoding="utf-8")


def test_pack_open_ignores_migration_sidecars(tmp_path: Path) -> None:
    """1 binding + N ``*.sha256sidecar.json``: open still selects the binding."""
    blob = blob_encode(_sample_dict())
    pack_dir = _write_mini_pack(tmp_path, [0, 1], [blob, blob])
    _write_sidecars(pack_dir)
    view = DevFeaturePackView.open(pack_dir, feature_level="F2")
    assert view.n_positions == 2
    assert view.positions.tolist() == [0, 1]


def test_pack_open_sidecars_only_still_refused(tmp_path: Path) -> None:
    """Sidecars but NO binding json: still a typed refusal (fail-closed)."""
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    _write_sidecars(pack_dir)
    with pytest.raises(DevPackError) as exc:
        DevFeaturePackView.open(pack_dir)
    assert exc.value.code == "PACK_BINDING_MISSING"


def test_pack_open_two_bindings_still_refused(tmp_path: Path) -> None:
    """Two real binding jsons (with sidecars present): still refused."""
    blob = blob_encode(_sample_dict())
    pack_dir = _write_mini_pack(tmp_path, [0], [blob])
    (pack_dir / ("f" * 64 + ".json")).write_text(
        json.dumps({"schema": "v53mp_dev_pack_v1"}), encoding="utf-8")
    _write_sidecars(pack_dir)
    with pytest.raises(DevPackError) as exc:
        DevFeaturePackView.open(pack_dir)
    assert exc.value.code == "PACK_BINDING_MISSING"


# ---------------------------------------------------------------------------
# trainer integration: pack vs none = bitwise-equal metrics
# ---------------------------------------------------------------------------
def test_trainer_pack_vs_none_metrics_identical(tmp_path: Path) -> None:
    """A tiny synthetic run with the pack equals the same run without it.

    Also pins that the no-pack path is unchanged (dev features built into
    the dev LRU exactly as before the pack existed).
    """
    torch = pytest.importorskip("torch")
    del torch
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        FEATURE_LEVEL_V53MP,
        train_run_v53mp,
    )
    from tests.pesn.test_r3_v53mp_trainer import _build_env, _smoke_cfg

    folds = {"cv0": 1, "cv1": 1, "cv2": 1, "cv_dev0": 0, "cv_dev1": 0}
    env = _build_env(tmp_path / "env", folds)
    dev_pos = env.dev_pos
    assert dev_pos

    # build a pack over the env's dev positions (the builder path, inline)
    blobs = []
    for pos in dev_pos:
        rec = env.ds.records([pos])[0]
        built = build_sample_features_v3(
            env.ds.to_sample_v3(rec), level=FEATURE_LEVEL_V53MP,
            intent_off=False)
        # mirror _FeatureLRU._finalize
        if built.get("y_teacher") is None:
            built = dict(built)
            built["y_teacher"] = np.zeros(
                (len(rec.intent.key_rows()), 3), dtype=np.float32)
        blobs.append(blob_encode(built))
    pack_dir = tmp_path / "pack"
    write_dev_feature_pack(
        pack_dir, dev_pos, blobs, feature_level=FEATURE_LEVEL_V53MP,
        intent_off=False, dataset_binding={"n_samples": len(env.ds)})

    cfg = _smoke_cfg(arm="A")
    out_none = tmp_path / "run_none"
    metrics_none = train_run_v53mp(
        env.ds, env.labels, env.endpoint, cfg, out_none,
        dev_positions=dev_pos, resume=False)
    out_pack = tmp_path / "run_pack"
    metrics_pack = train_run_v53mp(
        env.ds, env.labels, env.endpoint, cfg, out_pack,
        dev_positions=dev_pos, resume=False, dev_feature_pack=str(pack_dir))

    assert metrics_pack["dev_feature_pack"] is not None
    assert metrics_pack["dev_feature_pack"]["n_positions"] == len(dev_pos)
    assert metrics_none["dev_feature_pack"] is None
    # the eval-derived blocks must be EXACTLY equal (all floats bitwise)
    for block in ("move", "move_guard_vs_r3", "q_head", "ordering",
                  "endpoint_head", "tau_end_calibration"):
        a, b = metrics_none[block], metrics_pack[block]
        _assert_exact_equal(a, b, block)
    assert metrics_none["run_outcome"] == metrics_pack["run_outcome"]
    assert (metrics_none["best_inner_val_loss"]
            == metrics_pack["best_inner_val_loss"])
    assert metrics_none["last_train_loss"] == metrics_pack["last_train_loss"]
    # training itself is pack-independent (same draws, same losses) —
    # compare the logs with the wall-clock field stripped
    def _log_no_time(path: Path) -> list:
        rows = []
        for raw in path.read_text("utf-8").splitlines():
            if not raw.strip():
                continue
            doc = json.loads(raw)
            doc.pop("elapsed_s", None)
            rows.append(doc)
        return rows
    assert _log_no_time(out_none / "train_log.jsonl") == \
        _log_no_time(out_pack / "train_log.jsonl")


def test_trainer_pack_open_happens_before_training(tmp_path: Path) -> None:
    """A stale pack is refused BEFORE any training work (fail-closed)."""
    torch = pytest.importorskip("torch")
    del torch
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        train_run_v53mp,
    )
    from tests.pesn.test_r3_v53mp_trainer import _build_env, _smoke_cfg

    env = _build_env(tmp_path / "env", {"cv0": 1, "cv1": 1, "cv2": 1,
                                        "cv_dev0": 0, "cv_dev1": 0})
    # a directory with a drifted binding json
    bad_pack = tmp_path / "bad_pack"
    bad_pack.mkdir()
    (bad_pack / "deadbeef.json").write_text(json.dumps({
        "schema": "v53mp_dev_pack_v1",
        "features_sha256": "0" * 64,
        "positions_sha256": "x",
        "feature_level": "F2",
    }), encoding="utf-8")
    out = tmp_path / "run_bad"
    with pytest.raises(DevPackError) as exc:
        train_run_v53mp(
            env.ds, env.labels, env.endpoint, _smoke_cfg(arm="A"), out,
            dev_positions=env.dev_pos, resume=False,
            dev_feature_pack=str(bad_pack))
    assert exc.value.code == "PACK_FEATURES_DRIFT"
    # no training artifacts were written
    assert not (out / "train_log.jsonl").is_file()
    assert not (out / "checkpoint_best.pt").is_file()


def _assert_exact_equal(a, b, path: str) -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict), path
        assert set(a) == set(b), path
        for key in a:
            _assert_exact_equal(a[key], b[key], "{}.{}".format(path, key))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_exact_equal(x, y, "{}[{}]".format(path, i))
    elif isinstance(a, float) and isinstance(b, float):
        assert np.float64(a).tobytes() == np.float64(b).tobytes(), \
            (path, a, b)
    else:
        assert a == b, (path, a, b)
