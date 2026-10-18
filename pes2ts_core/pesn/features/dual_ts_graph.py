"""Exchange-symmetric dual-graph feature assembly + shared endpoint encoder
(WP3-A, plan v4 §I.7; spec frozen in ``dual_ts_spec``; v4.1 WP1 = observation-
only 5-channel inputs, ``pesn_dual_ts_features_v3``).

Assembles, per state, the ``pesn_dual_ts_features_v3`` observation inputs:

* node scalar block (per atom, 5 channels — exact frozen layout from
  :mod:`pes2ts_core.pesn.features.dual_ts_spec`):
  ``[z_id, grad_norm_scaled, delta_e_scaled, charge, multiplicity]``;
* equivariant vector channels: ``X`` (Å, unweighted centroid removed at
  consume time) and ``g̃ = g / s_g`` (one shared positive scale, all axes);
* dual endpoint topology: the union edge set of the TWO endpoint graphs
  under the SHARED atom indexing, with per-endpoint per-atom and per-edge
  features kept as SEPARATE endpoint blocks (bond orders never merged or
  sorted across endpoints), consumed by a SHARED torch encoder whose
  outputs are fused exchange-symmetrically.

Exchange-symmetric fusion (frozen rule, ``dual_ts_spec.FUSION_RULE``):
``fused = concat([h_a + h_b, max(h_a, h_b)])`` where ``h_a``/``h_b`` are
the SHARED encoder's embeddings of endpoint A/B.  Sum and elementwise max
are both invariant under the endpoint swap, so the fusion is too — while
every per-edge bond order stays attached to its own endpoint block.  There
is NO per-edge sorting across endpoints anywhere in this module (that
flattening loses endpoint association and is rejected at the contract
layer).

Changed-bond definition: a union edge is *changed* when its bond STATUS
differs between endpoints, where status = ``(present, order)`` — a bond
formed/broken between the endpoints OR with a different bond order is
changed.

The disputed-5 rule is applied by :func:`consumed_endpoint_graphs`:
reactions flagged ``eligibility.bond_authority == "disputed"`` consume the
deterministic covalent-radius geometry perception of their episode-start
(endpoint frame) coordinates (same radii/tolerance as the
``dual_ts_preprocess`` fallback); all others consume the authority
endpoint graphs.  The per-sample ``bond_source`` flag keeps the two
populations separable (WP4 sensitivity slice; never silently mixed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from torch import nn

from pes2ts_core.pesn.contracts.dual_ts import (
    BOUNDARY_TEACHER_ENDPOINT,
    DualEndpointTask,
    DualTSTeacher,
    EndpointBond,
    EndpointGraph,
)
from pes2ts_core.pesn.features.dual_ts_spec import (
    BOND_SOURCE_AUTHORITY,
    BOND_SOURCE_COVALENT_FALLBACK,
    NEAR_ZERO_ACTION_THRESHOLD_ANGSTROM,
    NODE_SCALAR_NAMES,
    declared_layout_blocks,
)
from pes2ts_core.pesn.features.geometry import (
    COVALENT_RADII as _RUNTIME_COVALENT_RADII,
    covalent_radius,
)

DISPUTED_BOND_AUTHORITY_FLAG: Final[str] = "disputed"

#: Covalent-radius single-link fallback perception (disputed-5 rule).  The
#: radii and tolerance are the SAME frozen values the offline
#: ``dual_ts_preprocess`` fallback uses (Cordero 2008 single-bond radii,
#: 0.45 Å tolerance) — mirrored here because the runtime feature layer must
#: not import the offline layer (product boundary); a drift-guard test
#: asserts the two tables stay identical.
COVALENT_BOND_TOL_ANGSTROM: Final[float] = 0.45
BOND_PERCEPTION_COVALENT: Final[str] = "covalent_radius_single_link_v0"


def covalent_radius_endpoint_graph(
    atomic_numbers: np.ndarray, coords: np.ndarray,
) -> EndpointGraph:
    """Deterministic covalent-radius single-link perception of one endpoint
    frame (the disputed-5 fallback; single bond order 1.0, stereo unknown).

    Same rule, radii (:data:`pes2ts_core.pesn.features.geometry.
    COVALENT_RADII`) and tolerance as the offline ``dual_ts_preprocess``
    fallback — kept byte-compatible by a drift-guard test.
    """
    z = np.asarray(atomic_numbers, dtype=int)
    x = np.asarray(coords, dtype=float)
    if x.shape != (int(z.size), 3):
        raise DualTSGraphError(
            "GEOM_SHAPE_INVALID",
            f"perception coordinates {x.shape} incompatible with "
            f"{int(z.size)} atoms")
    bonds: list[EndpointBond] = []
    for i in range(int(z.size)):
        ri = covalent_radius(int(z[i]))
        for j in range(i + 1, int(z.size)):
            rj = covalent_radius(int(z[j]))
            dist = float(np.linalg.norm(x[i] - x[j]))
            if dist <= ri + rj + COVALENT_BOND_TOL_ANGSTROM:
                bonds.append(EndpointBond(i=i, j=j, bond_order=1.0))
    maps = tuple(f"m{k + 1:04d}" for k in range(int(z.size)))
    graph = EndpointGraph(
        atomic_numbers=z.copy(), atom_map_ids=maps, bonds=tuple(bonds))
    graph.validate()
    return graph

#: Number of node-scalar channels (frozen layout; must match the spec).
N_NODE_SCALARS: Final[int] = declared_layout_blocks()["node_scalar"]
N_ENDPOINT_ATOM_FEATS: Final[int] = declared_layout_blocks()["endpoint_atom_feat"]
N_ENDPOINT_EDGE_FEATS: Final[int] = declared_layout_blocks()["endpoint_edge_feat"]


class DualTSGraphError(RuntimeError):
    """Typed dual-graph feature assembly refusal."""


# ---------------------------------------------------------------------------
# Disputed-5 rule: consumed endpoint graphs
# ---------------------------------------------------------------------------


def consumed_endpoint_graphs(
    task: DualEndpointTask, teacher: DualTSTeacher,
) -> tuple[EndpointGraph, EndpointGraph, str, str]:
    """Endpoint graphs CONSUMED as encoder input under the frozen disputed-5
    rule, with their per-side ``bond_source`` flags.

    * ``bond_authority == "disputed"`` → BOTH sides consume the
      covalent-radius perception (:func:`endpoint_graph_from_geometry`,
      same radii + 0.45 Å tolerance as the ``dual_ts_preprocess`` fallback)
      run on that branch's episode-start (endpoint frame) exported
      coordinates; authority bonds stay in the task record for bookkeeping.
    * otherwise → the authority endpoint graphs are consumed unchanged.

    Returns ``(graph_a, graph_b, source_a, source_b)`` where side "A" is
    endpoint index 0 (the canonical content-digest order fixed by the task
    record) and branch/episode identity comes from the teacher records.

    Truth-access allowlist (oracle G1 finding 6): the ONLY teacher data
    consumed here is episode-start geometry — ``episode.start_state_id`` →
    ``teacher.states[].coordinates``, the branch ``endpoint_label``, and
    Z/maps (from the task).  Teacher ``labels`` / ``action_delta`` /
    ``ts_displacement`` / ``remaining_arc_length`` are NEVER read into the
    endpoint graphs; endpoint graphs are task-contract input and the
    episode-start frame is the observable initial condition at t=0.
    """
    authority = str(task.eligibility.get("bond_authority", ""))
    z = np.asarray(task.endpoints[0].atomic_numbers, dtype=int)
    if authority == DISPUTED_BOND_AUTHORITY_FLAG:
        # Episode-start state of each branch = that endpoint's frame.
        start_coords: dict[str, np.ndarray] = {}
        for episode in teacher.episodes:
            state = next(
                (s for s in teacher.states
                 if s.state_id == episode.start_state_id), None)
            if state is None:
                raise DualTSGraphError(
                    "EPISODE_START_MISSING",
                    f"episode {episode.episode_id} references unknown start "
                    f"state {episode.start_state_id}")
            branch = next((b for b in teacher.branches
                           if b.branch_id == episode.branch_id), None)
            if branch is None:
                raise DualTSGraphError(
                    "BRANCH_MISSING",
                    f"episode {episode.episode_id} references unknown branch")
            if np.asarray(state.coordinates, dtype=float).shape != (int(z.size), 3):
                raise DualTSGraphError(
                    "START_COORDS_SHAPE_INVALID",
                    f"state {state.state_id} coordinates incompatible with "
                    f"{int(z.size)} atoms")
            start_coords[branch.endpoint_label] = np.asarray(
                state.coordinates, dtype=float)
        graphs: list[EndpointGraph] = []
        sources: list[str] = []
        for label in ("A", "B"):
            if label not in start_coords:
                raise DualTSGraphError(
                    "EPISODE_START_MISSING",
                    f"no episode-start state found for endpoint {label}")
            graphs.append(covalent_radius_endpoint_graph(
                z, start_coords[label]))
            sources.append(BOND_SOURCE_COVALENT_FALLBACK)
        return graphs[0], graphs[1], sources[0], sources[1]
    # Authority consumption (geometry_verified / accepted_with_note): the
    # task endpoint graphs are consumed exactly as registered.
    return (task.endpoints[0], task.endpoints[1],
            BOND_SOURCE_AUTHORITY, BOND_SOURCE_AUTHORITY)


# ---------------------------------------------------------------------------
# Dual topology features (numpy; shared atom indexing, endpoint blocks kept)
# ---------------------------------------------------------------------------


def _bond_maps(graph: EndpointGraph) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for bond in graph.bonds:
        i, j = int(bond.i), int(bond.j)
        if i > j:
            i, j = j, i
        out[(i, j)] = float(bond.bond_order)
    return out


def union_edge_pairs(
    bonds_a: dict[tuple[int, int], float],
    bonds_b: dict[tuple[int, int], float],
) -> np.ndarray:
    """Sorted (E, 2) union edge set ``bonds_a | bonds_b`` (pairs i<j)."""
    pairs = sorted(set(bonds_a) | set(bonds_b))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def endpoint_edge_features(
    bonds_own: dict[tuple[int, int], float],
    bonds_other: dict[tuple[int, int], float],
    union_pairs: np.ndarray,
) -> np.ndarray:
    """(E, 4) per-edge features of ONE endpoint on the union edge set:
    ``[bond_present, bond_order, changed_flag, stereo_known]``.

    ``changed_flag`` = 1 iff the bond STATUS (present, order) differs
    between the two endpoints — a symmetric property carried by both
    endpoint blocks.  Bond orders stay bound to their own endpoint block.
    """
    rows = np.zeros((int(union_pairs.shape[0]), N_ENDPOINT_EDGE_FEATS),
                    dtype=np.float64)
    for e, pair in enumerate(union_pairs):
        key = (int(pair[0]), int(pair[1]))
        own = bonds_own.get(key)
        other = bonds_other.get(key)
        if own is not None:
            rows[e, 0] = 1.0
            rows[e, 1] = float(own)
        status_own = None if own is None else float(own)
        status_other = None if other is None else float(other)
        rows[e, 2] = 1.0 if status_own != status_other else 0.0
    return rows


def endpoint_atom_features(
    n_atoms: int,
    bonds_own: dict[tuple[int, int], float],
    bonds_other: dict[tuple[int, int], float],
    union_pairs: np.ndarray,
) -> np.ndarray:
    """(N, 4) per-atom features of ONE endpoint:
    ``[degree, bond_order_sum, changed_bond_degree, stereo_known_degree]``.
    """
    degree = np.zeros(n_atoms)
    order_sum = np.zeros(n_atoms)
    changed = np.zeros(n_atoms)
    stereo = np.zeros(n_atoms)
    for e, pair in enumerate(union_pairs):
        i, j = int(pair[0]), int(pair[1])
        key = (i, j)
        own = bonds_own.get(key)
        other = bonds_other.get(key)
        if own is not None:
            degree[i] += 1.0
            degree[j] += 1.0
            order_sum[i] += float(own)
            order_sum[j] += float(own)
        changed_flag = (None if own is None else float(own)) != \
            (None if other is None else float(other))
        if changed_flag:
            changed[i] += 1.0
            changed[j] += 1.0
    feats = np.stack([degree, order_sum, changed, stereo], axis=1)
    return feats


@dataclass(frozen=True, slots=True)
class DualTopologyFeatures:
    """Dual endpoint topology for ONE reaction (shared atom indexing).

    Edge features are stored per ENDPOINT block on the directed expansion
    of the union edge set — never merged across endpoints.
    """

    reaction_id: str
    atomic_numbers: np.ndarray        # (N,) int
    edge_pairs: np.ndarray            # (E, 2) undirected union, sorted, i<j
    edge_index: np.ndarray            # (2, 2E) directed both ways
    atom_feats_a: np.ndarray          # (N, 4)
    atom_feats_b: np.ndarray          # (N, 4)
    edge_feats_a: np.ndarray          # (2E, 4)
    edge_feats_b: np.ndarray          # (2E, 4)
    bond_source_a: str
    bond_source_b: str

    @property
    def n_atoms(self) -> int:
        return int(self.atomic_numbers.size)

    @property
    def n_edges_undirected(self) -> int:
        return int(self.edge_pairs.shape[0])

    @property
    def n_edges_directed(self) -> int:
        return int(self.edge_index.shape[1])


def build_topology_features(
    *,
    task: DualEndpointTask,
    teacher: DualTSTeacher,
) -> DualTopologyFeatures:
    """Apply the disputed-5 rule and build the dual topology features.

    Observation-contract input only (oracle G1 finding 6): the ``teacher``
    record is read solely for episode-start geometry (``start_state_id``
    coordinates, Z, maps) and endpoint labels via
    :func:`consumed_endpoint_graphs`; teacher ``labels`` / ``action_delta`` /
    ``ts_displacement`` / ``remaining_arc_length`` are never read into any
    topology array (those are supervision, consumed only by
    :func:`build_teacher_supervision`).
    """
    graph_a, graph_b, source_a, source_b = consumed_endpoint_graphs(
        task, teacher)
    z = np.asarray(graph_a.atomic_numbers, dtype=int)
    if not np.array_equal(z, np.asarray(graph_b.atomic_numbers, dtype=int)):
        raise DualTSGraphError(
            "ENDPOINT_ATOM_MISMATCH",
            f"endpoint graphs of {task.reaction_id} disagree on Z")
    bonds_a = _bond_maps(graph_a)
    bonds_b = _bond_maps(graph_b)
    pairs = union_edge_pairs(bonds_a, bonds_b)
    # Directed expansion: for each undirected (i, j): i->j then j->i.
    directed = np.concatenate([pairs, pairs[:, ::-1]], axis=0)  # (2E, 2)
    edge_index = np.ascontiguousarray(directed.T)               # (2, 2E)
    ea = endpoint_edge_features(bonds_a, bonds_b, pairs)
    eb = endpoint_edge_features(bonds_b, bonds_a, pairs)
    edge_feats_a = np.concatenate([ea, ea], axis=0)
    edge_feats_b = np.concatenate([eb, eb], axis=0)
    return DualTopologyFeatures(
        reaction_id=str(task.reaction_id),
        atomic_numbers=z,
        edge_pairs=pairs,
        edge_index=edge_index,
        atom_feats_a=endpoint_atom_features(int(z.size), bonds_a, bonds_b, pairs),
        atom_feats_b=endpoint_atom_features(int(z.size), bonds_b, bonds_a, pairs),
        edge_feats_a=edge_feats_a,
        edge_feats_b=edge_feats_b,
        bond_source_a=str(source_a),
        bond_source_b=str(source_b),
    )


# ---------------------------------------------------------------------------
# Per-state input assembly (numpy)
# ---------------------------------------------------------------------------


def remove_translation(x: np.ndarray) -> np.ndarray:
    """Consume-time translation removal (frozen rule): unweighted centroid."""
    x = np.asarray(x, dtype=float)
    return x - x.mean(axis=0, keepdims=True)


@dataclass(frozen=True, slots=True)
class DualStateFeatures:
    """One state's assembled OBSERVATION inputs (numpy; exact frozen layout).

    Observation-only (v4.1 WP1): NO teacher-label-derived channels.  Teacher
    supervision (``near_zero_action_mask``, ``is_teacher_endpoint``) is built
    separately by :func:`build_teacher_supervision` and never enters the
    network input blocks.
    """

    state_id: str
    atomic_numbers: np.ndarray          # (N,) int
    node_scalars: np.ndarray            # (N, N_NODE_SCALARS) = (N, 5)
    x_centered: np.ndarray              # (N, 3) Å, centroid removed
    g_scaled: np.ndarray                # (N, 3) Hartree/Å
    delta_e_scaled: float


@dataclass(frozen=True, slots=True)
class DualTeacherSupervision:
    """Teacher-label-derived supervision for ONE state (offline only).

    NEVER a network input.  Consumed solely by supervision targets, loss
    masks, and offline eval metadata.
    """

    near_zero_action_mask: np.ndarray   # (N,) bool, True = masked OUT
    is_teacher_endpoint: bool


def build_state_features(
    *,
    state_id: str,
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    gradient: np.ndarray,
    delta_e_hartree: float,
    s_g: float,
    s_delta_e: float,
    charge: int,
    multiplicity: int,
) -> DualStateFeatures:
    """Assemble one state's OBSERVATION node inputs per the frozen spec.

    * ``grad_norm_scaled = ||g_i||_2 / s_g`` (one shared scale, all axes);
    * ``delta_e_scaled = ΔE / s_ΔE`` (broadcast scalar).

    Only on-line observable quantities are accepted: geometry, gradient,
    ΔE (bound to the episode start), charge, multiplicity.  Teacher-label-
    derived values (``action_delta``, ``is_teacher_endpoint``) are NOT
    accepted here — they are supervision, built by
    :func:`build_teacher_supervision`.
    """
    if s_g <= 0.0 or s_delta_e <= 0.0:
        raise DualTSGraphError(
            "SCALE_INVALID", "s_g and s_delta_e must be finite > 0")
    z = np.asarray(atomic_numbers, dtype=int)
    x_raw = np.asarray(coordinates, dtype=float)
    g = np.asarray(gradient, dtype=float).reshape(int(z.size), 3)
    if x_raw.shape != (int(z.size), 3):
        raise DualTSGraphError(
            "GEOM_SHAPE_INVALID",
            f"coordinates {x_raw.shape} incompatible with {int(z.size)} atoms")
    if not (np.isfinite(x_raw).all() and np.isfinite(g).all()):
        raise DualTSGraphError("NONFINITE_INPUT", "non-finite X or g")
    n = int(z.size)
    grad_norm_scaled = np.linalg.norm(g, axis=1) / float(s_g)
    delta_e_scaled = float(delta_e_hartree) / float(s_delta_e)
    node_scalars = np.stack([
        z.astype(np.float64),                                # z_id channel
        grad_norm_scaled,
        np.full(n, delta_e_scaled),
        np.full(n, float(charge)),
        np.full(n, float(multiplicity)),
    ], axis=1)
    if node_scalars.shape != (n, N_NODE_SCALARS):
        raise DualTSGraphError(
            "LAYOUT_MISMATCH",
            f"node scalar block {node_scalars.shape} != (N, "
            f"{N_NODE_SCALARS})")
    return DualStateFeatures(
        state_id=str(state_id),
        atomic_numbers=z,
        node_scalars=node_scalars,
        x_centered=remove_translation(x_raw),
        g_scaled=g / float(s_g),
        delta_e_scaled=delta_e_scaled,
    )


def build_teacher_supervision(
    *,
    action_delta: np.ndarray | None,
    is_teacher_endpoint: bool,
    n_atoms: int,
    near_zero_threshold: float = NEAR_ZERO_ACTION_THRESHOLD_ANGSTROM,
) -> DualTeacherSupervision:
    """Build the teacher-label-derived supervision for ONE state.

    Preserves the frozen rules EXACTLY (v4.1 WP1 split out of the
    observation builder):

    * terminal teacher endpoint → ``near_zero_action_mask`` all-True (no
      action exists);
    * non-terminal with ``action_delta is None`` → ``ACTION_MISSING`` typed
      refusal (same error identity as the pre-split builder);
    * else per-atom ``near_zero_action_mask[i] = ||a_i||_2 < threshold``.

    Supervision never enters the network: callers write it to shard npz keys
    and consume it in loss masks / offline eval only.
    """
    n = int(n_atoms)
    if action_delta is None:
        if not is_teacher_endpoint:
            raise DualTSGraphError(
                "ACTION_MISSING",
                "state has no action but is not a teacher endpoint "
                "(only terminal states may lack an action)")
        near_zero = np.ones(n, dtype=bool)
    else:
        a = np.asarray(action_delta, dtype=float).reshape(n, 3)
        if not np.isfinite(a).all():
            raise DualTSGraphError(
                "NONFINITE_INPUT", "non-finite action in supervision")
        near_zero = np.linalg.norm(a, axis=1) < float(near_zero_threshold)
    return DualTeacherSupervision(
        near_zero_action_mask=near_zero,
        is_teacher_endpoint=bool(is_teacher_endpoint),
    )


# ---------------------------------------------------------------------------
# Shared endpoint encoder + exchange-symmetric fusion (torch)
# ---------------------------------------------------------------------------


def fuse_endpoint_pair(h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
    """Frozen exchange-symmetric fusion ``concat([h_a + h_b, max(h_a, h_b)])``.

    Both operations are invariant under swapping (h_a, h_b), so the fused
    features are invariant under the endpoint swap; endpoint association of
    the underlying per-edge bond orders is untouched (each ``h`` comes from
    its own endpoint block through the SHARED encoder).
    """
    if h_a.shape != h_b.shape:
        raise DualTSGraphError(
            "FUSION_SHAPE_MISMATCH",
            f"endpoint embeddings {tuple(h_a.shape)} vs {tuple(h_b.shape)}")
    return torch.cat([h_a + h_b, torch.maximum(h_a, h_b)], dim=-1)


@dataclass(frozen=True, slots=True)
class DualTSEncoderConfig:
    """Encoder sizes (structural; this lane trains nothing)."""

    d_model: int = 32
    d_element: int = 16
    max_z: int = 32          # embedding table covers Z in 0..max_z
    # V4.2 E1 (features_v4): per-endpoint edge feats widen 4 -> 7 with the
    # three gradient-direction dot channels; default keeps v3 semantics.
    edge_feat_dim: int = N_ENDPOINT_EDGE_FEATS


class DualTSFusionEncoder(nn.Module):
    """SHARED per-endpoint encoder + exchange-symmetric fusion.

    The SAME atom/edge encoders (identical weights) embed each endpoint's
    per-atom and per-edge blocks; the two embeddings are fused by
    :func:`fuse_endpoint_pair`.  The encoder never sees an endpoint label —
    only content — so swapping the two endpoints' inputs swaps the two
    ``h`` tensors and leaves the fused output unchanged (verified by
    tests).  Equivariant vector channels (X, g̃) pass through OUTSIDE this
    module: scalars here are invariant by design, vectors are handled by
    the (later, WP3-B) equivariant layers.
    """

    def __init__(self, config: DualTSEncoderConfig | None = None):
        super().__init__()
        cfg = config or DualTSEncoderConfig()
        self.config = cfg
        self.element_embedding = nn.Embedding(cfg.max_z + 1, cfg.d_element)
        self.atom_encoder = nn.Sequential(
            nn.Linear(N_ENDPOINT_ATOM_FEATS + cfg.d_element, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(cfg.edge_feat_dim, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def encode_endpoint_atoms(
        self, z: torch.Tensor, atom_feats: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.element_embedding(z.long())
        return self.atom_encoder(torch.cat([atom_feats.float(), emb], dim=-1))

    def encode_endpoint_edges(self, edge_feats: torch.Tensor) -> torch.Tensor:
        return self.edge_encoder(edge_feats.float())

    def forward(
        self,
        *,
        z: torch.Tensor,
        atom_feats_a: torch.Tensor,
        atom_feats_b: torch.Tensor,
        edge_feats_a: torch.Tensor,
        edge_feats_b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h_a = self.encode_endpoint_atoms(z, atom_feats_a)
        h_b = self.encode_endpoint_atoms(z, atom_feats_b)
        he_a = self.encode_endpoint_edges(edge_feats_a)
        he_b = self.encode_endpoint_edges(edge_feats_b)
        return {
            "fused_atom": fuse_endpoint_pair(h_a, h_b),   # (N, 2 d_model)
            "fused_edge": fuse_endpoint_pair(he_a, he_b),  # (2E, 2 d_model)
            "h_atom_a": h_a,
            "h_atom_b": h_b,
            "h_edge_a": he_a,
            "h_edge_b": he_b,
        }


def topology_to_tensors(
    topo: DualTopologyFeatures,
) -> dict[str, torch.Tensor]:
    """Numpy topology -> torch tensors for :class:`DualTSFusionEncoder`."""
    return {
        "z": torch.from_numpy(np.ascontiguousarray(topo.atomic_numbers)).long(),
        "atom_feats_a": torch.from_numpy(
            np.ascontiguousarray(topo.atom_feats_a)).float(),
        "atom_feats_b": torch.from_numpy(
            np.ascontiguousarray(topo.atom_feats_b)).float(),
        "edge_feats_a": torch.from_numpy(
            np.ascontiguousarray(topo.edge_feats_a)).float(),
        "edge_feats_b": torch.from_numpy(
            np.ascontiguousarray(topo.edge_feats_b)).float(),
    }


__all__ = [
    "BOND_PERCEPTION_COVALENT",
    "COVALENT_BOND_TOL_ANGSTROM",
    "DISPUTED_BOND_AUTHORITY_FLAG",
    "DualStateFeatures",
    "DualTeacherSupervision",
    "DualTSEncoderConfig",
    "DualTSGraphError",
    "DualTopologyFeatures",
    "DualTSFusionEncoder",
    "N_ENDPOINT_ATOM_FEATS",
    "N_ENDPOINT_EDGE_FEATS",
    "N_NODE_SCALARS",
    "NODE_SCALAR_NAMES",
    "build_state_features",
    "build_teacher_supervision",
    "build_topology_features",
    "covalent_radius_endpoint_graph",
    "consumed_endpoint_graphs",
    "endpoint_atom_features",
    "endpoint_edge_features",
    "fuse_endpoint_pair",
    "remove_translation",
    "topology_to_tensors",
    "union_edge_pairs",
]
