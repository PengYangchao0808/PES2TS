"""Dual-TS feature spec v4 + gradient-augmented edge features + key-pair
conditioning (plan v4 §6, contract §1 E1/E2, ADR-0024 item 4).

This module is the Lane B write-domain surface for the E1/E2 feature
extensions.  It never modifies the frozen v3 modules — it *reuses* the
``dual_ts_spec`` rules/digest machinery and the ``dual_ts_graph`` topology
builders read-only, and adds:

* ``pesn_dual_ts_features_v4`` — the E1 feature spec: ``endpoint_edge_feat``
  grows 4 → 7 channels by appending three gradient-direction inner products
  (``rhat_ij·ghat_i``, ``rhat_ij·ghat_j``, ``ghat_i·ghat_j``) to the frozen
  ``[bond_present, bond_order, changed_flag, stereo_known]`` block.
  ``node_scalars`` stay 5-channel; the v4 digest records the v3 spec digest
  as its ``parent_spec_digest``.
* :func:`augment_edge_features_gradient` — the pure per-edge augmentation
  (numpy), returning the 7-channel block plus a per-atom zero-gradient-norm
  mask.
* :func:`select_key_pairs` + :func:`build_key_pair_records` +
  :class:`KeyPairConditioner` — the E2 sparse key-pair conditioning path
  (changed-bond whitelist only, deploy-time available), encoded per pair and
  broadcast-summed into a FiLM(γ, β) modulation of the node scalars ``h``.

Discipline: the key-pair encoding is built ONLY from the current geometry
``x`` and the union topology (changed-bond whitelist).  It is exchange-
symmetric under an endpoint A/B swap by construction (changed_flag is a
symmetric property of the union edge set) and never reads teacher labels.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import nn

from pes2ts_core.pesn.contracts.dual_ts import (
    EndpointGraph,
    content_digest_of,
)
from pes2ts_core.pesn.features.dual_ts_spec import (
    build_features_spec,
    declared_layout_blocks,
    save_spec_payload,
)
from pes2ts_core.pesn.features.dual_ts_graph import DualTSGraphError

#: E1 feature version (contract §1 identity).
DUAL_TS_FEATURES_V4_VERSION: Final[str] = "pesn_dual_ts_features_v4"

#: Number of channels in the v4 augmented per-edge feature block.
N_EDGE_FEATS_V4: Final[int] = 7

#: v4 per-edge feature names (the frozen 4-channel bond block + 3 gradient
#: direction inner products).
EDGE_FEAT_NAMES_V4: Final[tuple[str, ...]] = (
    "bond_present", "bond_order", "changed_flag", "stereo_known",
    "rhat_dot_ghat_i", "rhat_dot_ghat_j", "ghat_i_dot_ghat_j",
)

#: E2 key-pair record layout (see :func:`build_key_pair_records`).
KEY_PAIR_DIST_N_BASIS: Final[int] = 32
KEY_PAIR_DIST_CUTOFF: Final[float] = 5.0
#: dist RBF(32) + atom_i state(2) + atom_j state(2) + role(2) + kind/missing(2)
KEY_PAIR_RECORD_DIM: Final[int] = 32 + 2 + 2 + 2 + 2


# ---------------------------------------------------------------------------
# E1: gradient-augmented edge features (7-channel)
# ---------------------------------------------------------------------------


def augment_edge_features_gradient(
    edge_feats_4ch: np.ndarray,
    x: np.ndarray,
    g: np.ndarray,
    edge_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Append three gradient-direction inner products to a 4-channel edge block.

    ``edge_feats_4ch`` is the frozen per-edge ``[bond_present, bond_order,
    changed_flag, stereo_known]`` block (one endpoint's view, or any 4-channel
    edge block) on the DIRECTED edge expansion, shape ``(E, 4)``.  For each
    directed edge ``(src, dst)`` (``edge_index`` shape ``(2, E)``, bidirectional
    union edges), append:

    * ``rhat_ij · ghat_i``   where ``rhat_ij = (x_src − x_dst)/‖·‖``,
    * ``rhat_ij · ghat_j``,
    * ``ghat_i · ghat_j``,

    with ``ghat = g / ‖g‖`` per atom.  Atoms with ``‖g‖ == 0`` are zeroed
    (their ``ghat`` row is exactly zero) and reported in the returned mask.

    Returns ``(augmented, zero_norm_mask)`` where ``augmented`` has shape
    ``(E, 7)`` and ``zero_norm_mask`` is a boolean ``(N,)`` array that is
    ``True`` exactly for atoms whose gradient norm is zero (their ``ghat`` was
    zeroed).  Pure numpy — no torch, no state, no teacher data.
    """
    edge_feats = np.asarray(edge_feats_4ch, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    ei = np.asarray(edge_index, dtype=np.int64)
    if edge_feats.ndim != 2 or edge_feats.shape[1] != 4:
        raise DualTSGraphError(
            "EDGE_FEATS_SHAPE_INVALID",
            f"edge_feats_4ch must be (E, 4), got {edge_feats.shape}")
    if x.ndim != 2 or x.shape[1] != 3 or g.shape != x.shape:
        raise DualTSGraphError(
            "GEOM_SHAPE_INVALID",
            f"x {x.shape} / g {g.shape} must both be (N, 3)")
    if ei.ndim != 2 or ei.shape[0] != 2 or ei.shape[1] != edge_feats.shape[0]:
        raise DualTSGraphError(
            "EDGE_INDEX_SHAPE_INVALID",
            f"edge_index {ei.shape} incompatible with {edge_feats.shape[0]} edges")

    n = int(x.shape[0])
    grad_norm = np.linalg.norm(g, axis=1)              # (N,)
    zero_norm_mask = grad_norm == 0.0                  # (N,) bool
    ghat = np.zeros_like(g)
    nonzero = ~zero_norm_mask
    ghat[nonzero] = g[nonzero] / grad_norm[nonzero, None]

    src = ei[0]
    dst = ei[1]
    rel = x[src] - x[dst]                              # (E, 3)
    dist = np.linalg.norm(rel, axis=1)                 # (E,)
    rhat = np.zeros_like(rel)
    safe = dist > 0.0
    rhat[safe] = rel[safe] / dist[safe, None]

    ch_rhat_gi = np.einsum("ed,ed->e", rhat, ghat[src])     # rhat_ij·ghat_i
    ch_rhat_gj = np.einsum("ed,ed->e", rhat, ghat[dst])     # rhat_ij·ghat_j
    ch_gi_gj = np.einsum("ed,ed->e", ghat[src], ghat[dst])  # ghat_i·ghat_j

    augmented = np.concatenate([
        edge_feats,
        ch_rhat_gi[:, None],
        ch_rhat_gj[:, None],
        ch_gi_gj[:, None],
    ], axis=1)
    if augmented.shape != (edge_feats.shape[0], N_EDGE_FEATS_V4):
        raise DualTSGraphError(
            "LAYOUT_MISMATCH",
            f"augmented edge block {augmented.shape} != "
            f"(E, {N_EDGE_FEATS_V4})")
    return augmented, zero_norm_mask


# ---------------------------------------------------------------------------
# E2: key-pair whitelist (changed-bond union edges) + pair records
# ---------------------------------------------------------------------------


def _bond_map(graph: EndpointGraph) -> dict[tuple[int, int], float]:
    """Undirected ``{(i, j): bond_order}`` map of one endpoint (i < j)."""
    out: dict[tuple[int, int], float] = {}
    for bond in graph.bonds:
        i, j = int(bond.i), int(bond.j)
        if i > j:
            i, j = j, i
        out[(i, j)] = float(bond.bond_order)
    return out


def select_key_pairs(
    topology_a: EndpointGraph, topology_b: EndpointGraph,
) -> np.ndarray:
    """The E2 key-pair whitelist: union edges whose bond STATUS differs
    between the two endpoints (``changed_flag == 1``).

    Deploy-time available: it reads only the two endpoint graphs (task-contract
    input), never teacher labels or geometry.  Returns a ``(K, 2)`` int array
    of ``(i, j)`` pairs with ``i < j``; empty ``(0, 2)`` when nothing changed.
    """
    bonds_a = _bond_map(topology_a)
    bonds_b = _bond_map(topology_b)
    union = sorted(set(bonds_a) | set(bonds_b))
    changed: list[tuple[int, int]] = []
    for key in union:
        # changed_flag = 1 iff the bond STATUS (present, order) differs.
        if bonds_a.get(key) != bonds_b.get(key):
            changed.append(key)
    if not changed:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(changed, dtype=np.int64).reshape(-1, 2)


def _gaussian_basis(dist: float, cutoff: float, n_basis: int) -> np.ndarray:
    """v1-reused Gaussian expansion of a scalar distance (numpy)."""
    centers = np.linspace(0.0, cutoff, n_basis)
    scale = cutoff / n_basis
    return np.exp(-0.5 * ((dist - centers) / scale) ** 2)


def build_key_pair_records(
    *,
    key_pairs: np.ndarray,
    x: np.ndarray,
    topology_a: EndpointGraph,
    topology_b: EndpointGraph,
    n_basis: int = KEY_PAIR_DIST_N_BASIS,
    cutoff: float = KEY_PAIR_DIST_CUTOFF,
) -> np.ndarray:
    """Build the E2 per-pair conditioning records ``(K, KEY_PAIR_RECORD_DIM)``.

    Each record (all from the current geometry + union topology, exchange-
    symmetric under an endpoint A/B swap):

    * ``dist_rbf`` (32): v1 Gaussian basis of ``‖x_i − x_j‖`` (Å);
    * atom ``i`` bond state (2): ``[changed_degree_i, union_degree_i]``;
    * atom ``j`` bond state (2): ``[changed_degree_j, union_degree_j]``;
    * role bits (2): ``[driver_i, driver_j]`` — ``driver_k = 1`` iff atom ``k``
      is incident to a changed bond (both are for a changed edge; guards — atoms
      merely adjacent to a change — are the non-driver neighbours);
    * kind/missing (2): ``[changed_kind, missing_flag]`` where ``changed_kind``
      is ``0.0`` (order changed, present in both endpoints) or ``1.0``
      (formed/broken, present in exactly one) and ``missing_flag = 0.0``
      (no missing masks in this wave).

    ``union_degree`` / ``changed_degree`` are counted over the UNION edge set so
    the record is identical under swapping ``topology_a``/``topology_b``.
    """
    key_pairs = np.asarray(key_pairs, dtype=np.int64).reshape(-1, 2)
    x = np.asarray(x, dtype=np.float64)
    bonds_a = _bond_map(topology_a)
    bonds_b = _bond_map(topology_b)
    union = sorted(set(bonds_a) | set(bonds_b))
    n = int(x.shape[0])

    union_degree = np.zeros(n)
    changed_degree = np.zeros(n)
    for (i, j) in union:
        union_degree[i] += 1.0
        union_degree[j] += 1.0
        if bonds_a.get((i, j)) != bonds_b.get((i, j)):
            changed_degree[i] += 1.0
            changed_degree[j] += 1.0

    records = np.zeros((int(key_pairs.shape[0]), KEY_PAIR_RECORD_DIM))
    for r, (i, j) in enumerate(key_pairs):
        i, j = int(i), int(j)
        dist = float(np.linalg.norm(x[i] - x[j]))
        rbf = _gaussian_basis(dist, float(cutoff), int(n_basis))
        status_a = bonds_a.get((i, j))
        status_b = bonds_b.get((i, j))
        changed_kind = 0.0 if (status_a is not None and status_b is not None) else 1.0
        records[r] = np.concatenate([
            rbf,                                     # (32,)
            [changed_degree[i], union_degree[i]],    # atom i state (2,)
            [changed_degree[j], union_degree[j]],    # atom j state (2,)
            [1.0 if changed_degree[i] >= 1 else 0.0,
             1.0 if changed_degree[j] >= 1 else 0.0],  # role (2,)
            [changed_kind, 0.0],                     # kind/missing (2,)
        ])
    return records


class KeyPairConditioner(nn.Module):
    """E2 sparse key-pair conditioner: pair records → per-node broadcast-sum →
    FiLM(γ, β) modulation of ``h``.

    ``forward(h, pair_records, pair_indices)``:

    1. encode each pair record with an MLP ``pair_mlp`` (``pair_hidden``);
    2. broadcast-sum the pair embeddings onto their two endpoint atoms;
    3. FiLM-modulate the node scalars ``h`` via ``γ, β = film_gamma, film_beta``
       (identity at init: γ→1, β→0) and return ``h * γ + β``.

    No key pairs → the identity map (returns ``h`` unchanged).  All parameters
    are ordinary registered parameters and are counted by ``n_parameters``.
    """

    def __init__(self, hidden: int, record_dim: int = KEY_PAIR_RECORD_DIM,
                 pair_hidden: int = 64) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.record_dim = int(record_dim)
        self.pair_hidden = int(pair_hidden)
        self.pair_mlp = nn.Sequential(
            nn.Linear(self.record_dim, self.pair_hidden), nn.SiLU(),
            nn.Linear(self.pair_hidden, self.pair_hidden))
        self.film_gamma = nn.Linear(self.pair_hidden, self.hidden)
        self.film_beta = nn.Linear(self.pair_hidden, self.hidden)
        # Identity FiLM at init: γ = 1, β = 0.
        nn.init.ones_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, h: torch.Tensor, pair_records: torch.Tensor,
                pair_indices: torch.Tensor) -> torch.Tensor:
        n = int(h.shape[0])
        if int(pair_records.shape[0]) == 0:
            return h
        emb = self.pair_mlp(pair_records.to(h.dtype))       # (K, pair_hidden)
        cond = torch.zeros(n, self.pair_hidden, device=h.device, dtype=h.dtype)
        idx = pair_indices.long()
        cond = cond.index_add(0, idx[:, 0], emb)
        cond = cond.index_add(0, idx[:, 1], emb)
        gamma = self.film_gamma(cond)                       # (N, hidden)
        beta = self.film_beta(cond)                         # (N, hidden)
        return h * gamma + beta


# ---------------------------------------------------------------------------
# Spec v4 (reuses the v3 digest machinery; parent digest bound)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DualTSFeaturesSpecV4:
    """Lightweight holder for the ``pesn_dual_ts_features_v4`` payload.

    The v4 schema is NOT a ``DualTSFeaturesSpec`` instance (that contract class
    is frozen to v3/legacy-v2); it reuses the SAME canonical-JSON content digest
    (:func:`content_digest_of`) so the v4 identity is comparable and stable.
    """

    payload: dict
    schema_version: str = DUAL_TS_FEATURES_V4_VERSION

    def to_payload(self) -> dict:
        return dict(self.payload)

    def content_digest(self) -> str:
        return content_digest_of(self.payload)


def build_features_spec_v4(
    *,
    s_g: float,
    s_delta_e: float,
    element_vocab: list[int],
    population: dict,
    n_atoms_range: tuple[int, int],
) -> tuple[DualTSFeaturesSpecV4, dict]:
    """Materialize the ``pesn_dual_ts_features_v4`` spec with a v3 parent digest.

    Reuses the frozen v3 rules/values by building the v3 spec first (same
    scales/population) and binding its content digest as ``parent_spec_digest``.
    The v4 ``declared_layout`` sets ``endpoint_edge_feat = 7`` (the augmented
    block); ``node_scalar`` stays 5.  Returns ``(spec_v4, payload)``.
    """
    v3_spec, v3_payload = build_features_spec(
        s_g=s_g, s_delta_e=s_delta_e, element_vocab=list(element_vocab),
        population=population, n_atoms_range=n_atoms_range)
    parent_digest = v3_spec.content_digest()

    layout = declared_layout_blocks()
    v4_layout = dict(layout)
    v4_layout["endpoint_edge_feat"] = N_EDGE_FEATS_V4

    support_domain = dict(v3_payload["support_domain"])
    support_domain["freeze_version"] = DUAL_TS_FEATURES_V4_VERSION
    support_domain["parent_schema_version"] = v3_payload["schema_version"]
    support_domain["edge_feature_augmentation"] = {
        "version": "pesn_dual_ts_features_v4_edge_augmentation_v1",
        "edge_feat_names": list(EDGE_FEAT_NAMES_V4),
        "gradient_direction": (
            "ghat = g / ||g||; zero-norm atoms zeroed + masked; "
            "directed edges computed independently"),
    }

    payload: dict = {
        "schema_version": DUAL_TS_FEATURES_V4_VERSION,
        "parent_spec_digest": parent_digest,
        "encoding": v3_payload["encoding"],
        "equivariant_vectors": list(v3_payload["equivariant_vectors"]),
        "delta_energy_definition": v3_payload["delta_energy_definition"],
        "frozen_scales": dict(v3_payload["frozen_scales"]),
        "masks": dict(v3_payload["masks"]),
        "support_domain": support_domain,
        "declared_layout": v4_layout,
        "input_dim": int(sum(v4_layout.values())),
    }
    spec = DualTSFeaturesSpecV4(payload=payload)
    # Digestion also validates the payload is canonical-JSON serializable.
    spec.content_digest()
    return spec, payload


def load_spec_v4_payload(path: str | Path) -> tuple[DualTSFeaturesSpecV4, dict]:
    """Load a saved v4 spec payload, verifying its recorded content digest.

    Mirrors ``dual_ts_spec.load_spec_payload`` but does NOT instantiate a v3
    ``DualTSFeaturesSpec`` (the v4 schema is separate).
    """
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = dict(record["spec"])
    digest = content_digest_of(payload)
    if digest != str(record["spec_digest"]):
        raise ValueError(
            f"features spec v4 digest mismatch in {path}: recorded "
            f"{record['spec_digest']!r} != recomputed {digest!r}")
    return DualTSFeaturesSpecV4(payload=payload), payload


__all__ = [
    "DUAL_TS_FEATURES_V4_VERSION",
    "EDGE_FEAT_NAMES_V4",
    "KEY_PAIR_DIST_CUTOFF",
    "KEY_PAIR_DIST_N_BASIS",
    "KEY_PAIR_RECORD_DIM",
    "N_EDGE_FEATS_V4",
    "DualTSFeaturesSpecV4",
    "KeyPairConditioner",
    "augment_edge_features_gradient",
    "build_features_spec_v4",
    "build_key_pair_records",
    "load_spec_v4_payload",
    "select_key_pairs",
]
