"""B2 acceptance: scope-before-access in the single audited truth bridge."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pes2ts_core.g0.truth import truth_reader
from pes2ts_core.pesn.offline.data import truth_source
from pes2ts_core.pesn.offline.splits.freeze import (
    SplitFreezeError,
    TRAIN_DEV,
    TRAIN_FIT,
    freeze_split,
    write_split_manifest,
)
from pes2ts_core.utils.jsonio import write_json

RID_TRAIN = "RXN_0000000001"
RID_SEALED = "RXN_0000000500"


@pytest.fixture()
def split_tree(tmp_path: Path) -> Path:
    interim = tmp_path / "interim"
    interim.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "reaction_id": [RID_TRAIN, RID_SEALED],
                "split": ["train", "test"],
            }
        ),
        interim / "split_assignment.parquet",
    )
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    write_json(manifests / "split_manifest.json", {"schema_version": "g0_manifest_v1"})
    write_json(
        interim / "duplicate_groups.json",
        {"schema_version": "g0_manifest_v1", "dataset_version": "f", "groups": []},
    )
    manifest = freeze_split(
        interim,
        manifests,
        seed=42,
        family_by_reaction={RID_TRAIN: "fa", RID_SEALED: "fb"},
    )
    # force RID_TRAIN into train_fit for the bridge test
    manifest["subsets"][TRAIN_FIT] = [RID_TRAIN]
    manifest["subsets"][TRAIN_DEV] = []
    return write_split_manifest(manifest, tmp_path / "splits").parent


def test_bridge_refuses_without_allow_truth(split_tree: Path, tmp_path: Path) -> None:
    with pytest.raises(truth_source.TruthBridgeError):
        truth_source.load_scoped_irc_frames(
            RID_TRAIN,
            allow_truth=False,
            manifests_dir=tmp_path / "manifests",
            split_dir=split_tree,
        )


def test_bridge_refuses_sealed_reactions_before_any_read(
    split_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _must_not_read(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("accessor must not be reached for sealed reactions")

    monkeypatch.setattr(truth_reader, "load_irc_frames", _must_not_read)
    with pytest.raises(truth_source.TruthBridgeError):
        truth_source.load_scoped_irc_frames(
            RID_SEALED,
            allow_truth=True,
            manifests_dir=tmp_path / "manifests",
            split_dir=split_tree,
        )


def test_bridge_passes_scope_and_records_caller(
    split_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[object, ...]] = []

    def _fake_load(reaction_id, allow_truth, *, manifests_dir, caller=None):
        calls.append((reaction_id, allow_truth, caller))
        return {"reaction_id": reaction_id, "coordinates": [], "n_atoms": 0, "n_frames": 0}

    monkeypatch.setattr(truth_reader, "load_irc_frames", _fake_load)
    frames = truth_source.load_scoped_irc_frames(
        RID_TRAIN,
        allow_truth=True,
        manifests_dir=tmp_path / "manifests",
        split_dir=split_tree,
    )
    assert frames["reaction_id"] == RID_TRAIN
    assert calls == [(RID_TRAIN, True, truth_source.TRUTH_BRIDGE_MODULE)]


def test_only_the_bridge_module_is_truth_allowed_in_pesn() -> None:
    from pes2ts_core.utils.truth_guard import DEFAULT_ALLOWLIST

    pesn_allowed = [
        entry for entry in DEFAULT_ALLOWLIST if entry.startswith("pes2ts_core/pesn/")
    ]
    assert pesn_allowed == ["pes2ts_core/pesn/offline/data/truth_source.py"]


def test_guarded_pesn_modules_carry_no_guard_markers() -> None:
    # Same markers the static guard itself flags, restricted to the pesn tree.
    pesn_root = Path(truth_source.__file__).resolve().parents[2]
    offenders: list[str] = []
    for module in sorted(pesn_root.rglob("*.py")):
        if module.name == "truth_source.py":
            continue
        source = module.read_text(encoding="utf-8")
        for marker in ("ground_truth", "truth_sources", "B3LYPD3_TZVP.h5", "B3LYPD3_TZVP_IRC.h5"):
            if marker in source:
                offenders.append(f"{module.name}: {marker}")
    assert not offenders, f"truth markers leaked into guarded modules: {offenders}"


def test_missing_split_manifest_refuses(tmp_path: Path) -> None:
    with pytest.raises((truth_source.TruthBridgeError, SplitFreezeError)):
        truth_source.load_scoped_irc_frames(
            RID_TRAIN,
            allow_truth=True,
            manifests_dir=tmp_path,
            split_dir=tmp_path / "nonexistent",
        )
