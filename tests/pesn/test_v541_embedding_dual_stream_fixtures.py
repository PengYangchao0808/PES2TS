"""Shared fixtures/helpers for the V5.4.1 WP3b E5 ``dual_stream`` tests
(plan §6.7 E5; registry family ``dual_stream``, conditional row on
E1/E4).

Two disciplines are enforced here (identical to the E1–E4 fixture
modules):

* **lazy candidate import** — ``dual_stream`` registers the real E5
  builder AT IMPORT TIME (the WP3 path), so the test modules must NOT
  import it at collection time: that would leak the registration into
  unrelated WP2 registry tests running in the same session.
  :func:`load_e5` imports on demand and is always paired with the
  caller's ``registry_snapshot``/``registry_restore`` (each test file
  wraps it in an autouse fixture).  Importing ``dual_stream`` also
  transitively imports the E1/E4 modules (it reuses their operator
  classes) — their registrations are the same real WP3 builders and are
  covered by the same snapshot/restore;
* **genuine F2 forward batches** — every batch comes from the WP3a
  fixture module (real legacy pipeline), so the F2-masked first 12
  ``edge_attr`` slots are zero and the E5 tests exercise exactly the
  inputs ``model.forward`` will feed the branch.

Also provided: controlled-weight builders that collapse the 48→24
projections, the GINE24 / Angle24 streams and the fusion to a
hand-computable pipeline, plus the ``j/k`` side-swap view transform.
"""

from __future__ import annotations

import dataclasses
import importlib

import torch

from pes2ts_core.pesn.models.v541.embedding_registry import register_embedding
from tests.pesn.test_v541_embedding_chemical_fixtures import (
    assert_all_floats,
    assert_close,
    patch_count_only_helpers,
    random_h,
)
from tests.pesn.test_v541_feature_providers_fixtures import (
    angle_geometry_batch,
    bonded_pair_batch,
    dense_sphere_batch,
    extreme_coordinate_batch,
    forward_batch,
    make_intent,
    make_sample,
    no_neighbor_batch,
    one_chem_one_spatial_batch,
    single_atom_batch,
    sn2_bidirectional_batch,
    sn2_forward_batch,
    sn2_reverse_batch,
    sphere_center_batch,
    stretched_bond_batch,
    zero_length_batch,
)

#: fp32 tolerances for isometry/batch comparisons of the E5 output.
ABS_TOL = 1e-5
REL_TOL = 1e-5

_E5_MODULE = "pes2ts_core.pesn.models.v541.embeddings.dual_stream"

#: Frozen §6.7 E5 branch width.
BRANCH_WIDTH = 24

#: Exact §6.7 E5 branch parameter count (no external W/alpha):
#: two 48→24 projections (2·1176) + GINE24 (1416) + Angle24 (5928)
#: + fusion 48→48 (2352).
E5_BRANCH_PARAMS_CANDIDATE = 12_048


def load_e5():
    """Import the E5 module (registers ``E5``) and return it.

    Callers MUST snapshot/restore the registry around their test — the
    import side effect must never outlive the test that asked for it.
    """
    module = importlib.import_module(_E5_MODULE)
    # Idempotent re-registration: a previous test's registry_restore may
    # have removed the entry while the module stays cached in sys.modules.
    register_embedding("E5", module.build_dual_stream, replace=True)
    return module


def dual_stream_batch_matrix() -> dict[str, dict]:
    """The §6.4/§6.8 fixture matrix E5 is tested against (union of the
    chemical and geometric matrices — E5 consumes BOTH views)."""
    return {
        "single_atom": single_atom_batch(),
        "bonded_pair": bonded_pair_batch(),
        "no_neighbor": no_neighbor_batch(),
        "stretched": stretched_bond_batch(),
        "sn2_forward": sn2_forward_batch(),
        "sn2_reverse": sn2_reverse_batch(),
        "sn2_bidir": sn2_bidirectional_batch(n_graphs=2),
        "one_chem_one_spatial": one_chem_one_spatial_batch(),
        "shell3": sphere_center_batch([1.0, 1.0, 1.0]),
        "shell_tie": sphere_center_batch([1.0] * 11 + [2.0] * 3),
        "dense": dense_sphere_batch(15),
        "angle_right": angle_geometry_batch("right"),
        "angle_pi": angle_geometry_batch("pi"),
        "angle_zero": angle_geometry_batch("zero"),
        "zero_length": zero_length_batch(),
        "extreme_scaled": extreme_coordinate_batch(1e4),
        "extreme_huge": extreme_coordinate_batch(1.0, huge=1e17),
    }


def chain_batch() -> dict:
    """3-atom chain 0–1 (1.0 Å) –2 (5.0 Å from 1): G_chem = the two
    bonds; centre 1 has exactly one angle (neighbours {0, 2})."""
    sample = make_sample(
        (6, 1, 1),
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (5.0, 0.0, 0.0)],
        make_intent(((0, 1, 1.0), (1, 2, 1.0)), ((0, 1, 1.0), (1, 2, 1.0)),
                    cv_deltas=(0.1,), cv_coords=((0, 1),)))
    return forward_batch([sample])


# --- controlled weights (hand-computable math at width 24) ------------------
def controlled_projections(module) -> None:
    """``chem_in`` selects ``h[:, :24]``, ``angle_in`` selects
    ``h[:, 24:]`` — the two streams see DISJOINT 24-wide slices, so every
    hand value below is unambiguous."""
    with torch.no_grad():
        w = torch.zeros(BRANCH_WIDTH, 48)
        w[:, :BRANCH_WIDTH] = torch.eye(BRANCH_WIDTH)
        module.chem_in.weight.copy_(w)
        module.chem_in.bias.zero_()
        w2 = torch.zeros(BRANCH_WIDTH, 48)
        w2[:, BRANCH_WIDTH:] = torch.eye(BRANCH_WIDTH)
        module.angle_in.weight.copy_(w2)
        module.angle_in.bias.zero_()


def controlled_gine24(module) -> None:
    """GINE24: ``P_e = 0`` and ``out_mlp = ReLU`` (identity layers).

    With positive inputs: ``f_c_i = h_c_i + Σ_j h_c_j`` (explicit self
    term + neighbour sum).
    """
    with torch.no_grad():
        module.pe_proj.weight.zero_()
        module.pe_proj.bias.zero_()
        module.out_mlp[0].weight.copy_(torch.eye(BRANCH_WIDTH))
        module.out_mlp[0].bias.zero_()
        module.out_mlp[2].weight.copy_(torch.eye(BRANCH_WIDTH))
        module.out_mlp[2].bias.zero_()


def controlled_angle24(module) -> None:
    """Angle24: ``P_e = 0``, ``a_ij = h_a_j`` (pair MLP selects the
    ``h_j`` block), triplet message ``= a_ij + a_ik`` (triplet MLP
    selects the pair-sum block), ``out_mlp = ReLU``.

    With positive inputs: ``f_a_i = Σ_{{j,k}} (h_a_j + h_a_k)`` for angle
    centres, exactly ``0`` elsewhere (mask).
    """
    with torch.no_grad():
        module.pe_proj.weight.zero_()
        module.pe_proj.bias.zero_()
        pair_w = torch.zeros(BRANCH_WIDTH, 2 * BRANCH_WIDTH + 12)
        pair_w[:, :BRANCH_WIDTH] = torch.eye(BRANCH_WIDTH)
        module.pair_mlp[0].weight.copy_(pair_w)
        module.pair_mlp[0].bias.zero_()
        module.pair_mlp[2].weight.copy_(torch.eye(BRANCH_WIDTH))
        module.pair_mlp[2].bias.zero_()
        trip_w = torch.zeros(BRANCH_WIDTH, 3 * BRANCH_WIDTH + 4)
        trip_w[:, BRANCH_WIDTH:2 * BRANCH_WIDTH] = torch.eye(BRANCH_WIDTH)
        module.triplet_mlp[0].weight.copy_(trip_w)
        module.triplet_mlp[0].bias.zero_()
        module.triplet_mlp[2].weight.copy_(torch.eye(BRANCH_WIDTH))
        module.triplet_mlp[2].bias.zero_()
        module.out_mlp[0].weight.copy_(torch.eye(BRANCH_WIDTH))
        module.out_mlp[0].bias.zero_()
        module.out_mlp[2].weight.copy_(torch.eye(BRANCH_WIDTH))
        module.out_mlp[2].bias.zero_()


def controlled_fusion(module, *, block: str = "identity") -> None:
    """Fusion selector: ``identity`` (48→48 eye), ``chem`` (keep the
    ``f_c`` block: out[:24] = in[:24], everything else 0), or ``angle``
    (keep the ``f_a`` block: out[24:] = in[24:]).  Zero bias in every
    mode."""
    with torch.no_grad():
        w = torch.zeros(48, 48)
        if block == "identity":
            w.copy_(torch.eye(48))
        elif block == "chem":
            w[:BRANCH_WIDTH, :BRANCH_WIDTH] = torch.eye(BRANCH_WIDTH)
        elif block == "angle":
            w[BRANCH_WIDTH:, BRANCH_WIDTH:] = torch.eye(BRANCH_WIDTH)
        else:                                    # pragma: no cover
            raise AssertionError(f"unknown fusion block {block!r}")
        module.fusion.weight.copy_(w)
        module.fusion.bias.zero_()


def controlled_e5(module) -> None:
    """Fully controlled E5: disjoint 24-slices, GINE24 = self+neighbour
    sum, Angle24 = pair-sum aggregate, fusion = identity."""
    controlled_projections(module)
    controlled_gine24(module.chem_branch)
    controlled_angle24(module.angle_branch)
    controlled_fusion(module, block="identity")


def swap_sides(view):
    """The same triplet view with the unordered pair's columns exchanged."""
    return dataclasses.replace(
        view,
        side_index=view.side_index.flip(1),
        side_distance=view.side_distance.flip(1),
        side_rbf12=view.side_rbf12.flip(1),
        side_chem_attr=view.side_chem_attr.flip(1),
    )


# --- shared sanity (does not depend on registry global state) --------------
def test_shared_batches_are_f2_forward_views() -> None:
    """Every shared batch is a genuine forward view with the F2 RBF mask."""
    from pes2ts_core.pesn.contracts.key_atom_v3 import FORWARD_VIEW_KEYS_V3

    for name, batch in dual_stream_batch_matrix().items():
        assert set(batch) <= set(FORWARD_VIEW_KEYS_V3), name
        edge_attr = batch["edge_attr"]
        assert bool((edge_attr[:, :12] == 0).all()), f"{name}: F2 slots"
        assert edge_attr.shape[1] == 17, name


__all__ = [
    "ABS_TOL",
    "BRANCH_WIDTH",
    "E5_BRANCH_PARAMS_CANDIDATE",
    "REL_TOL",
    "assert_all_floats",
    "assert_close",
    "chain_batch",
    "controlled_angle24",
    "controlled_e5",
    "controlled_fusion",
    "controlled_gine24",
    "controlled_projections",
    "dual_stream_batch_matrix",
    "load_e5",
    "no_neighbor_batch",
    "one_chem_one_spatial_batch",
    "patch_count_only_helpers",
    "random_h",
    "sn2_bidirectional_batch",
    "sn2_forward_batch",
    "sphere_center_batch",
    "swap_sides",
]
