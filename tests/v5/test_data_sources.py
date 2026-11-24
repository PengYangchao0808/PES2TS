"""Dataset-source contract: synthetic fixture, V5.5 store, real ``idx_v2:<path>``.

The real-data path is wired through the data package
(``pes2ts_data.readers.idx_v2.IdxV2Reader``) with lazy imports; on a host where
the idx_v2 store or the Zone-2 readers are absent, the engine must raise the
typed errors (``DATA_SOURCE_UNREACHABLE`` / ``DATA_CONTRACT_UNAVAILABLE``) instead
of fabricating data.  ``pes2ts_v5.data.idx_v2`` must stay unreferenced.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

from pes2ts_v5.configs.schema import config_to_dict, load_experiment_config
from pes2ts_v5.data.contracts import (
    DataContractUnavailable,
    DataSourceUnreachable,
)
from pes2ts_v5.data.resolve import (
    SOURCE_IDX_V2,
    SOURCE_STORE,
    SOURCE_SYNTHETIC,
    parse_store_spec,
    resolve_dataset,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "experiments"
V5_PACKAGE = Path(__file__).resolve().parents[2] / "pes2ts_v5"


def _doc(tmp_path: Path, *, store_manifest=None, policy="synthetic_fixture") -> dict:
    doc = config_to_dict(load_experiment_config(CONFIG_DIR / "smoke.yaml"))
    if store_manifest is not None:
        doc["data"]["store_manifest"] = store_manifest
    doc["data"]["split_policy"] = policy
    return doc


def _readers_available() -> bool:
    try:
        import pes2ts_data.readers.idx_v2  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Source parsing
# ---------------------------------------------------------------------------

def test_parse_store_spec_variants() -> None:
    assert parse_store_spec("idx_v2:/pes2ts_data/v53mp_50k_data/idx_v2") == (
        SOURCE_IDX_V2, "/pes2ts_data/v53mp_50k_data/idx_v2")
    assert parse_store_spec("IDX_V2:data/store") == (SOURCE_IDX_V2, "data/store")
    assert parse_store_spec("data/interim/v5_synthetic") == (
        SOURCE_STORE, "data/interim/v5_synthetic")


# ---------------------------------------------------------------------------
# Synthetic fixture + V5.5 store unchanged
# ---------------------------------------------------------------------------

def test_synthetic_fixture_resolution_unchanged(tmp_path, write_yaml) -> None:
    doc = _doc(tmp_path, store_manifest=str(tmp_path / "synthetic"))
    cfg = load_experiment_config(write_yaml(doc))
    dataset, info = resolve_dataset(cfg)
    assert info["source"] == SOURCE_SYNTHETIC
    assert info["dims"] == {"frame_feat_dim": 11, "static_feat_dim": 8, "q_dim": 3}
    assert dataset.counts["n_reactions"] > 0
    batch, metas = next(iter(dataset.iter_batches(
        "train", batch_size=2, shuffle=False, seed=17, max_batches=1)))
    assert batch["frame_feats"].shape[0] == 2
    assert metas and metas[0]["trajectory_id"]


def test_existing_store_manifest_resolution(tmp_path, write_yaml, synthetic_store) -> None:
    doc = _doc(tmp_path, store_manifest=str(synthetic_store))
    cfg = load_experiment_config(write_yaml(doc))
    dataset, info = resolve_dataset(cfg)
    assert info["source"] == SOURCE_STORE
    assert info["dims"]["frame_feat_dim"] == 11


# ---------------------------------------------------------------------------
# Real idx_v2 source: typed failures on this host
# ---------------------------------------------------------------------------

def test_idx_v2_missing_path_is_typed_unreachable(tmp_path, write_yaml) -> None:
    doc = _doc(tmp_path, store_manifest="idx_v2:/pes2ts_data/v53mp_50k_data/idx_v2",
               policy="idx_v2_real")
    cfg = load_experiment_config(write_yaml(doc))
    with pytest.raises(DataSourceUnreachable) as excinfo:
        resolve_dataset(cfg)
    assert excinfo.value.code == "DATA_SOURCE_UNREACHABLE"
    assert "does not exist" in str(excinfo.value)


def test_idx_v2_directory_without_curves_json_is_typed_unreachable(tmp_path,
                                                                  write_yaml) -> None:
    empty = tmp_path / "idx_empty"
    empty.mkdir()
    doc = _doc(tmp_path, store_manifest=f"idx_v2:{empty}", policy="idx_v2_real")
    cfg = load_experiment_config(write_yaml(doc))
    with pytest.raises(DataSourceUnreachable) as excinfo:
        resolve_dataset(cfg)
    assert excinfo.value.code == "DATA_SOURCE_UNREACHABLE"
    assert "curves.json" in str(excinfo.value)


def test_idx_v2_present_but_readers_absent_is_typed_contract_unavailable(
        tmp_path, write_yaml) -> None:
    """On this host Zone-2 readers are not on disk yet -> typed contract error."""
    index = tmp_path / "idx"
    index.mkdir()
    (index / "curves.json").write_text("{}", encoding="utf-8")
    doc = _doc(tmp_path, store_manifest=f"idx_v2:{index}", policy="idx_v2_real")
    cfg = load_experiment_config(write_yaml(doc))
    expected = ("DATA_SOURCE_UNREACHABLE" if _readers_available()
                else "DATA_CONTRACT_UNAVAILABLE")
    with pytest.raises((DataContractUnavailable, DataSourceUnreachable)) as excinfo:
        resolve_dataset(cfg)
    assert excinfo.value.code == expected


def test_idx_v2_with_empty_path_is_typed(tmp_path, write_yaml) -> None:
    doc = _doc(tmp_path, store_manifest="idx_v2:", policy="idx_v2_real")
    cfg = load_experiment_config(write_yaml(doc))
    with pytest.raises(DataSourceUnreachable) as excinfo:
        resolve_dataset(cfg)
    assert excinfo.value.code == "DATA_SOURCE_UNREACHABLE"


def test_idx_v2_source_is_never_fabricated_from_synthetic_policy(tmp_path,
                                                                 write_yaml) -> None:
    """``idx_v2:`` + a synthetic split policy still refuses to fabricate data."""
    doc = _doc(tmp_path,
               store_manifest=f"idx_v2:{tmp_path / 'missing_idx'}",
               policy="synthetic_fixture")
    cfg = load_experiment_config(write_yaml(doc))
    with pytest.raises(DataSourceUnreachable):
        resolve_dataset(cfg)
    assert not (tmp_path / "missing_idx").exists()


# ---------------------------------------------------------------------------
# No private idx_v2 module import surface
# ---------------------------------------------------------------------------

def test_no_pes2ts_v5_idx_v2_import_surface() -> None:
    pattern = re.compile(r"(from|import)\s+pes2ts_v5\.data\.idx_v2\b")
    offenders: list[str] = []
    for path in sorted(V5_PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if path.name == "idx_v2.py" and path.parent.name == "data":
            continue  # legacy module pending the Lead's move into pes2ts_data
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(V5_PACKAGE)))
    assert not offenders, f"legacy idx_v2 import surface remains: {offenders}"


def test_config_accepts_idx_v2_store_manifest(tmp_path, write_yaml) -> None:
    doc = _doc(tmp_path, store_manifest="idx_v2:data/store/idx_v2",
               policy="idx_v2_real")
    cfg = load_experiment_config(write_yaml(doc))
    assert cfg.data.store_manifest.startswith("idx_v2:")
    # split_policy is free-form; the source kind is carried by the prefix
    assert copy.deepcopy(cfg).data.store_manifest == "idx_v2:data/store/idx_v2"
