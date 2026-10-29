"""Shared fixtures/helpers for the V5.4.1 WP3b geometric-candidate tests
(E3 ``schnet_spatial`` / E4 ``angle_mpnn``; plan §6.7–§6.8).

Two disciplines are enforced here:

* **lazy candidate import** — ``schnet_spatial`` / ``angle_mpnn`` register
  their real E3/E4 builders AT IMPORT TIME (the WP3 path), so the test
  modules must NOT import them at collection time: that would leak the
  registration into unrelated WP2 registry tests running in the same
  session.  :func:`load_e3` / :func:`load_e4` import on demand and are
  always paired with the caller's ``registry_snapshot``/``registry_restore``
  (each test file wraps them in an autouse fixture);
* **genuine F2 forward batches** — every batch comes from the WP3a
  fixture module (real legacy pipeline), so the F2-masked first 12
  ``edge_attr`` slots are zero and the E3/E4 geometry tests exercise
  exactly the inputs ``model.forward`` will feed the branch.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from pes2ts_core.pesn.contracts.key_atom_v3 import FORWARD_VIEW_KEYS_V3
from pes2ts_core.pesn.models.v541.embedding_registry import register_embedding
from tests.pesn.test_v541_feature_providers_fixtures import (
    angle_geometry_batch,
    bonded_pair_batch,
    dense_sphere_batch,
    extreme_coordinate_batch,
    no_neighbor_batch,
    one_chem_one_spatial_batch,
    single_atom_batch,
    sn2_bidirectional_batch,
    sn2_forward_batch,
    sphere_center_batch,
    stretched_bond_batch,
    zero_length_batch,
)

#: fp32 tolerances for isometry/batch comparisons of a CANDIDATE output
#: (row order of a scatter can differ between builds — exact equality is
#: only asserted where the reduction order provably cannot change: the
#: j/k swap and the chunking equivalence).
ABS_TOL = 1e-5
REL_TOL = 1e-5

_E3_MODULE = "pes2ts_core.pesn.models.v541.embeddings.schnet_spatial"
_E4_MODULE = "pes2ts_core.pesn.models.v541.embeddings.angle_mpnn"


def load_e3():
    """Import the E3 module (registers ``E3``) and return it.

    Callers MUST snapshot/restore the registry around their test — the
    import side effect must never outlive the test that asked for it.
    """
    module = importlib.import_module(_E3_MODULE)
    # Idempotent re-registration: a previous test's registry_restore may
    # have removed the entry while the module stays cached in sys.modules.
    register_embedding("E3", module.build_schnet_spatial, replace=True)
    return module


def load_e4():
    """Import the E4 module (registers ``E4``) and return it."""
    module = importlib.import_module(_E4_MODULE)
    register_embedding("E4", module.build_angle_mpnn, replace=True)
    return module


def random_h(n_nodes: int, seed: int = 0) -> torch.Tensor:
    """Deterministic synthetic ``h_legacy [n, 48]`` (fp32)."""
    gen = torch.Generator().manual_seed(int(seed))
    return torch.randn(int(n_nodes), 48, generator=gen)


def geometric_batch_matrix() -> dict[str, dict]:
    """The §6.4/§6.8 fixture matrix both candidates are tested against."""
    return {
        "single_atom": single_atom_batch(),
        "bonded_pair": bonded_pair_batch(),
        "no_neighbor": no_neighbor_batch(),
        "stretched": stretched_bond_batch(),
        "sn2_forward": sn2_forward_batch(),
        "sn2_bidir": sn2_bidirectional_batch(n_graphs=2),
        "one_chem_one_spatial": one_chem_one_spatial_batch(),
        "shell_tie": sphere_center_batch([1.0] * 11 + [2.0] * 3),
        "shell_over_cap": sphere_center_batch([1.5] * 13),
        "dense": dense_sphere_batch(15),
        # degenerate angle geometries: straight / near-straight / zero-angle
        "angle_right": angle_geometry_batch("right"),
        "angle_pi": angle_geometry_batch("pi"),
        "angle_near_pi": angle_geometry_batch("near_pi"),
        "angle_zero": angle_geometry_batch("zero"),
        "zero_length": zero_length_batch(),
        "extreme_scaled": extreme_coordinate_batch(1e4),
        "extreme_huge": extreme_coordinate_batch(1.0, huge=1e17),
    }


def assert_close(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
    assert a.shape == b.shape, what
    if a.numel() == 0:
        return
    diff = float((a - b).detach().abs().max())
    assert torch.allclose(a.detach(), b.detach(), rtol=REL_TOL, atol=ABS_TOL), (
        f"{what}: max abs diff {diff}")


def assert_all_floats(mapping: dict, what: str) -> None:
    for key, value in mapping.items():
        assert isinstance(value, float) and not isinstance(value, bool), (
            f"{what}[{key!r}] is {type(value).__name__}, expected float")
        assert value == value and value not in (float("inf"),), (
            f"{what}[{key!r}] = {value} is not finite")


def patch_count_only_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch every count-only host-sync helper to RAISE.

    Mirrors the WP3a proof: with ``diagnostics=False`` the candidates
    must never route through ``count_graphs`` / ``diagnostic_sum`` (the
    ``monkeypatch`` fixture undoes the patches at teardown).
    """
    from pes2ts_core.pesn.models.v541.feature_providers import (
        angle_triplets as angle_module,
    )
    from pes2ts_core.pesn.models.v541.feature_providers import (
        common as provider_common,
    )
    from pes2ts_core.pesn.models.v541.feature_providers import (
        legal_neighbors as legal_module,
    )

    def boom(*_args, **_kwargs):
        raise AssertionError("count-only helper called on the cheap path")

    monkeypatch.setattr(provider_common, "count_graphs", boom)
    monkeypatch.setattr(provider_common, "diagnostic_sum", boom)
    monkeypatch.setattr(legal_module, "diagnostic_sum", boom)
    monkeypatch.setattr(angle_module, "diagnostic_sum", boom)


# --- shared sanity (does not depend on registry global state) --------------
def test_shared_batches_are_f2_forward_views() -> None:
    """Every shared batch is a genuine forward view with the F2 RBF mask."""
    for name, batch in geometric_batch_matrix().items():
        assert set(batch) <= set(FORWARD_VIEW_KEYS_V3), name
        edge_attr = batch["edge_attr"]
        assert bool((edge_attr[:, :12] == 0).all()), f"{name}: F2 slots"
        assert edge_attr.shape[1] == 17, name
