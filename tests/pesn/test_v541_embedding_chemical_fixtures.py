"""Shared fixtures/helpers for the V5.4.1 WP3b chemical-candidate tests
(E1 ``gin_e_chemical`` / E2 ``pna_chemical``; plan §6.7–§6.8).

Two disciplines are enforced here:

* **lazy candidate import** — ``gin_e_chemical`` / ``pna_chemical``
  register their real E1/E2 builders AT IMPORT TIME (the WP3 path), so
  the test modules must NOT import them at collection time: that would
  leak the registration into unrelated WP2 registry tests running in the
  same session.  :func:`load_e1` / :func:`load_e2` import on demand and
  are always paired with the caller's ``registry_snapshot``/
  ``registry_restore`` (each test file wraps them in an autouse fixture);
* **genuine F2 forward batches** — every batch comes from the WP3a
  fixture module (real legacy pipeline), so the F2-masked first 12
  ``edge_attr`` slots are zero and the E1/E2 tests exercise exactly the
  inputs ``model.forward`` will feed the branch.

Also provided: controlled-weight builders (hand-computable aggregations),
the direction-flip / duplicate-edge batch transforms of §6.8 row 2, and
the count-only helper patch used to prove the ``diagnostics=False`` cheap
path.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from pes2ts_core.pesn.models.v541.embedding_registry import register_embedding
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

#: fp32 tolerances for isometry/batch comparisons of a CANDIDATE output
#: (row order of a scatter can differ between builds — exact equality is
#: only asserted where the reduction order provably cannot change).
ABS_TOL = 1e-5
REL_TOL = 1e-5

_E1_MODULE = "pes2ts_core.pesn.models.v541.embeddings.gin_e_chemical"
_E2_MODULE = "pes2ts_core.pesn.models.v541.embeddings.pna_chemical"

#: E2 readout block layout (frozen concat order of §6.7): for each
#: aggregator in (mean, max, min, std) × scaler in (identity,
#: amplification, attenuation), then the explicit self block last.
E2_BLOCK_INDEX: dict[str, int] = {
    f"{agg}x{scaler}": i
    for i, (agg, scaler) in enumerate(
        (a, s) for a in ("mean", "max", "min", "std")
        for s in ("identity", "amplification", "attenuation"))
}
E2_BLOCK_INDEX["self"] = 12


def load_e1():
    """Import the E1 module (registers ``E1``) and return it.

    Callers MUST snapshot/restore the registry around their test — the
    import side effect must never outlive the test that asked for it.
    """
    module = importlib.import_module(_E1_MODULE)
    # Idempotent re-registration: a previous test's registry_restore may
    # have removed the entry while the module stays cached in sys.modules.
    register_embedding("E1", module.build_gin_e_chemical, replace=True)
    return module


def load_e2():
    """Import the E2 module (registers ``E2``) and return it."""
    module = importlib.import_module(_E2_MODULE)
    register_embedding("E2", module.build_pna_chemical, replace=True)
    return module


def random_h(n_nodes: int, seed: int = 0) -> torch.Tensor:
    """Deterministic synthetic ``h_legacy [n, 48]`` (fp32, positive)."""
    gen = torch.Generator().manual_seed(int(seed))
    return torch.rand(int(n_nodes), 48, generator=gen) + 0.5


def chemical_batch_matrix() -> dict[str, dict]:
    """The §6.4/§6.8 fixture matrix both candidates are tested against."""
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
        "zero_length": zero_length_batch(),
        "extreme_scaled": extreme_coordinate_batch(1e4),
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


# --- batch transforms (§6.8 row 2: direction / duplicate edges) -------------
_EDGE_ALIGNED = ("edge_attr", "b_src", "b_tgt", "delta_b_intent",
                 "abs_delta", "is_changed")


def flip_edge_directions(batch: dict) -> dict:
    """Swap ``edge_index`` rows — every message direction is reversed.

    The view is bidirectional with bitwise-identical attributes, so the
    rebuilt ``G_chem`` (and every candidate delta) must be unchanged:
    message direction never flips the signed bond delta (§6.6.3).
    """
    out = dict(batch)
    out["edge_index"] = batch["edge_index"].flip(0)
    return out


def duplicate_first_edge(batch: dict) -> dict:
    """Append an EXACT duplicate of edge column 0 (all edge-aligned keys).

    The provider merges bitwise-equal duplicates (§6.6.3), so the
    candidate delta must be identical to the unduplicated batch.
    """
    out = dict(batch)
    out["edge_index"] = torch.cat(
        [batch["edge_index"], batch["edge_index"][:, :1]], dim=1)
    for key in _EDGE_ALIGNED:
        value = batch[key]
        out[key] = torch.cat([value, value[:1]], dim=0)
    return out


def poison_first_edge_attr(batch: dict) -> dict:
    """Duplicate edge 0 with an INCONSISTENT attribute (mismatch).

    The provider refuses ``V541_PROVIDER_EDGE_ATTR_MISMATCH`` — the
    candidates must propagate the typed refusal instead of silently
    merging.  The poisoned field is the ``b_src`` COLUMN (chem field 0);
    ``edge_attr`` RBF slots are F2-masked and unread.
    """
    out = duplicate_first_edge(batch)
    b_src = out["b_src"].clone()
    b_src[-1] = b_src[-1] + 1.0               # duplicate row now differs
    out["b_src"] = b_src
    return out


# --- controlled weights (hand-computable math) ------------------------------
def controlled_gine(module) -> None:
    """E1: ``P_e = 0`` and ``MLP = ReLU`` (identity layers).

    With positive ``h`` the whole pipeline becomes hand-computable:
    ``f_i = ReLU(h_i + Σ_j ReLU(h_j)) = h_i + Σ_j h_j``.
    """
    with torch.no_grad():
        module.pe_proj.weight.zero_()
        module.pe_proj.bias.zero_()
        module.out_mlp[0].weight.copy_(torch.eye(48))
        module.out_mlp[0].bias.zero_()
        module.out_mlp[2].weight.copy_(torch.eye(48))
        module.out_mlp[2].bias.zero_()


def controlled_pna(module, *, block: str = "meanxidentity") -> None:
    """E2: ``P_e = 0``, message MLP → ``h_i + h_j``, readout selects ONE
    concat block (default ``mean×identity``) with an identity output.

    With positive ``h``: ``m_ji = h_i + h_j`` and the selected block is
    emitted verbatim (ReLU is inert on non-negative values), so every
    aggregator/scaler block can be checked by hand.
    """
    k = E2_BLOCK_INDEX[block]
    with torch.no_grad():
        module.pe_proj.weight.zero_()
        module.pe_proj.bias.zero_()
        # msg MLP: [h_i | h_j | 0] → h_i + h_j (positive ⇒ ReLU inert)
        w1 = torch.zeros(48, 144)
        w1[:, :48] = torch.eye(48)
        w1[:, 48:96] = torch.eye(48)
        module.msg_mlp[0].weight.copy_(w1)
        module.msg_mlp[0].bias.zero_()
        module.msg_mlp[2].weight.copy_(torch.eye(48))
        module.msg_mlp[2].bias.zero_()
        # readout: select block k
        w = torch.zeros(48, 13 * 48)
        w[:, k * 48:(k + 1) * 48] = torch.eye(48)
        module.readout[0].weight.copy_(w)
        module.readout[0].bias.zero_()
        module.readout[2].weight.copy_(torch.eye(48))
        module.readout[2].bias.zero_()


def chain_batch() -> dict:
    """3-atom chain 0–1 (1.0 Å) –2 (5.0 Å from 1): G_chem = the two
    bonds; the 0–2 pair is neither covalent nor spatial."""
    sample = make_sample(
        (6, 1, 1),
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (5.0, 0.0, 0.0)],
        make_intent(((0, 1, 1.0), (1, 2, 1.0)), ((0, 1, 1.0), (1, 2, 1.0)),
                    cv_deltas=(0.1,), cv_coords=((0, 1),)))
    return forward_batch([sample])


def isolated_plus_bonded_batch() -> dict:
    """Atoms 0–1 bonded, atom 2 isolated (10 Å away, no bond).

    G_chem has exactly one pair → node 2 must aggregate ZERO chem
    messages (self only) while 0/1 see each other.
    """
    sample = make_sample(
        (6, 8, 1),
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        make_intent(((0, 1, 1.0),), ((0, 1, 1.0),),
                    cv_deltas=(0.1,), cv_coords=((0, 1),)))
    return forward_batch([sample])


# --- shared sanity (does not depend on registry global state) --------------
def test_shared_batches_are_f2_forward_views() -> None:
    """Every shared batch is a genuine forward view with the F2 RBF mask."""
    from pes2ts_core.pesn.contracts.key_atom_v3 import FORWARD_VIEW_KEYS_V3

    for name, batch in chemical_batch_matrix().items():
        assert set(batch) <= set(FORWARD_VIEW_KEYS_V3), name
        edge_attr = batch["edge_attr"]
        assert bool((edge_attr[:, :12] == 0).all()), f"{name}: F2 slots"
        assert edge_attr.shape[1] == 17, name
