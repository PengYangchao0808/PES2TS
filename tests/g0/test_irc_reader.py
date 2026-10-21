"""G0 IRC reader tests (V5.3-MP S0-B enabling change: energies/atomic_numbers).

The B3LYPD3_TZVP IRC archive stores per-frame energies under the dataset
name ``energies`` (not ``EHG``) and a per-reaction ``atomic_numbers``
vector.  ``read_irc_frames`` must surface both (additive, backwards
compatible: ``EHG`` keeps priority when both exist).
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from pes2ts_core.g0.irc_reader import read_irc_frames_at


def _write_archive(path, *, energy_name="energies", with_z=True):
    rng = np.random.RandomState(7)
    coords = rng.normal(size=(5, 4, 3)).astype("float32")
    energies = np.linspace(-100.0, -100.5, 5)
    with h5py.File(path, "w") as handle:
        grp = handle.create_group("RXN_0000000001")
        grp.create_dataset("coordinates", data=coords)
        grp.create_dataset(energy_name, data=energies)
        grp.create_dataset("forces", data=np.zeros_like(coords))
        if with_z:
            grp.create_dataset(
                "atomic_numbers", data=np.asarray([6, 7, 8, 1], dtype="int8"))
    return coords, energies


def test_read_irc_frames_energies_alias_and_atomic_numbers(tmp_path):
    path = tmp_path / "irc.h5"
    coords, energies = _write_archive(path)
    frames = read_irc_frames_at(path, "RXN_0000000001")
    assert frames is not None
    assert frames.n_frames == 5 and frames.n_atoms == 4
    assert frames.ehg is not None
    np.testing.assert_allclose(frames.ehg, energies)
    assert frames.atomic_numbers == (6, 7, 8, 1)
    assert frames.has_forces is True
    np.testing.assert_allclose(frames.coordinates, coords, atol=1e-7)


def test_read_irc_frames_ehg_keeps_priority(tmp_path):
    path = tmp_path / "irc.h5"
    coords, _ = _write_archive(path)
    with h5py.File(path, "a") as handle:
        handle["RXN_0000000001"].create_dataset(
            "EHG", data=np.full((5, 3), -1.0))
    frames = read_irc_frames_at(path, "RXN_0000000001")
    assert frames is not None and frames.ehg is not None
    assert frames.ehg.shape == (5, 3)
    np.testing.assert_allclose(frames.ehg, -1.0)


def test_read_irc_frames_missing_optional_datasets(tmp_path):
    path = tmp_path / "irc.h5"
    _write_archive(path, with_z=False)
    # remove energies entirely
    import h5py as h5
    with h5.File(path, "a") as handle:
        del handle["RXN_0000000001"]["energies"]
    frames = read_irc_frames_at(path, "RXN_0000000001")
    assert frames is not None
    assert frames.ehg is None
    assert frames.atomic_numbers is None


def test_read_irc_frames_absent_reaction_returns_none(tmp_path):
    path = tmp_path / "irc.h5"
    _write_archive(path)
    assert read_irc_frames_at(path, "RXN_0000000002") is None


def test_read_irc_frames_wrong_length_atomic_numbers_ignored(tmp_path):
    path = tmp_path / "irc.h5"
    coords, _ = _write_archive(path)
    import h5py as h5
    with h5.File(path, "a") as handle:
        grp = handle["RXN_0000000001"]
        del grp["atomic_numbers"]
        grp.create_dataset("atomic_numbers", data=np.asarray([6, 6]))
    frames = read_irc_frames_at(path, "RXN_0000000001")
    assert frames is not None
    assert frames.atomic_numbers is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
