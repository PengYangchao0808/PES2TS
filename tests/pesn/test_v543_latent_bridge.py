"""V5.4.3 WP2 — ``LatentBridge`` / ``DecoderContextV1`` CPU certificate.

Plan §5.2 (field table + 形状约束), §5.2.1 (两个互斥只读视图 + 硬隔离), §5.3
(``key_v`` 8 通道, 手算逐通道), §5.4 (``pair_s`` 48 维逐段), §5.5 (key 图边集
/ ``Ek ≤ 2(P+4K)`` / token 布局), §5.6 (禁止输入 + ``assert_no_supervision`` +
pair 集合来源审计), §5.7 (缓存键), F2 (从 forward edges 重构 changed K-pair,
规范序 + 符号 ΔB, 监督只用于离线比对), §11.5 (``test_v543_decoder_no_xyz`` /
``test_v543_no_supervision_leak`` / ``test_v543_track_view_isolation`` /
``test_v543_rotation_equivariance`` / ``test_v543_translation_invariance`` /
``test_v543_permutation_equivariance`` / ``test_v543_pair_index_ownership``).

Headline guarantees under test:

* the changed K-pair set reconstructed from the forward edges equals the
  supervision ``pair_index`` set — and the supervision tensor is passed as a
  SEPARATE argument to :func:`audit_pair_sets`, never into the Bridge (the
  in-test ``build_context(packet, batch_with_supervision)`` call is asserted
  to be REFUSED, which is the structural proof);
* pairs are canonical (key-slot ascending) and ``ΔB`` carries the sign;
* every §5.2 symmetry column holds: rotation → ``key_v``/``u0`` rotate,
  ``pair_s``/``key_edge_attr``/``key_s``/``graph_s`` invariant; translation →
  everything geometric invariant; atom/key permutation → rows permute with
  the documented covariance;
* the ``key_v`` 8 channels match a HAND-COMPUTED expectation (channels
  0/1/2/3/4/5/6/7 each value, including the zero-denominator → zero-vector
  rule of §5.3 :372);
* the ``pair_s`` 48 dims match a HAND-COMPUTED expectation, segment by
  segment, including ``RBF12`` from the frozen ``gaussian_rbf_v3`` centres,
  the endpoint-role blocks, the graph broadcast (padded ``cv_delta[0:6]``) and
  the fixed sketch ``W_fixed @ (key_s[i]-key_s[j])``;
* empty sets are first-class: ``P = 0`` and ``Ek = 0`` both build and
  validate; ``Ek ≤ 2(P + 4K)`` holds on every build;
* supervision keys are refused with a typed refusal; ``assert_no_supervision``
  passes on every built context, and a sentinel injected into the batch never
  reaches a context field (§5.6 :429);
* the two track views are hard-isolated with typed ``AttributeError``
  refusals (``V543_TRACK_ISOLATION``);
* a synthetic ``CoarsePacket`` (encoder output shape) integrates: every
  context field has the plan's shape/dtype/schema, the prefix sums close, and
  the Bridge stays parameter free (``0`` trainable, ``w_fixed`` not in
  ``state_dict``).

Fixture-level only, CPU-fast; the 10,000-frame real-data audit lives in
``tests/pesn/test_v543_bridge_pair_audit_realdata.py`` (``realdata`` marker).
"""

from __future__ import annotations

import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.features.key_atom_spec_v3 import BOND_SCALE  # noqa: E402
from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    EDGE_ABS_DELTA_KEY,
    EDGE_ATTR_KEY,
    EDGE_B_SRC_KEY,
    EDGE_B_TGT_KEY,
    EDGE_DELTA_B_KEY,
    EDGE_INDEX_KEY,
    EDGE_IS_CHANGED_KEY,
    EDGE_PACK_SLOTS_V3,
    FORWARD_VIEW_KEYS_V3,
    GRAPH_CV_DELTA_KEY,
    GRAPH_SIGNED_INTENT_KEY,
    NODE_GRAPH_KEY,
    NODE_INTENT_ROLE_KEY,
    NODE_KEY_SET_KEY,
    NODE_POS_KEY,
    NODE_Z_KEY,
    SUPERVISION_KEYS_V3,
    forward_view_v3,
)
from pes2ts_core.pesn.models.v543 import (  # noqa: E402
    A_REF_V543,
    BRIDGE_SPEC_SHA256,
    CACHE_KEY_FIELDS,
    DECODER_CONTEXT_FIELDS,
    DECODER_CONTEXT_IDENTITY_FIELDS,
    DECODER_CONTEXT_SCHEMA_V543,
    DECODER_CONTEXT_TENSOR_FIELDS,
    ENCODER_Z_DIM_V543,
    GRAPH_S_DIM_V543,
    KEY_EDGE_ATTR_DIM_V543,
    KEY_S_DIM_V543,
    KEY_V_CHANNELS_V543,
    KEY_V_CHANNEL_NAMES,
    MOVE_CONTEXT_FIELDS,
    MOVE_FORBIDDEN_FIELDS,
    PAIR_AUDIT_SCHEMA,
    PAIR_S_DIM_V543,
    PAIR_S_LAYOUT,
    PRECISION_VERSION_V543,
    PROCESS_CONTEXT_FIELDS,
    PROCESS_FORBIDDEN_FIELDS,
    CoarsePacket,
    DecoderContextV1,
    LatentBridge,
    MoveContextV1,
    ProcessContextV1,
    V543BridgeError,
    V543ContextError,
    V543SupervisionLeakError,
    V543TrackIsolationError,
    assert_no_supervision,
    audit_pair_sets,
    bridge_spec_sha256,
    build_context,
    cache_key,
    context_field_names,
    fixed_sketch_matrix,
    source_frame_identity,
    validate_decoder_context,
)
from scripts.agent_ops.feature_gate.v3_fixtures import (  # noqa: E402
    make_v3_batch,
    make_v3_gate_batch,
)

SEED = 17


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _node_u0(view: dict, *, seed: int = SEED) -> torch.Tensor:
    """Per-NODE ``u0`` fixture data ``[N,3]`` — key-order independent.

    ``u0``'s contract source is ``DirectMoveDecoder`` (plan §5.2 table 来源
    column), i.e. an ENCODER output and a **polar vector** in Å.  Building it
    per-node makes a key permutation a pure row re-indexing, and lets a
    rotation test rotate the encoder output with the geometry (which is what
    an E(3)-equivariant decoder would emit), instead of mis-attributing
    ``u0``'s symmetry to the Bridge.
    """
    generator = torch.Generator().manual_seed(seed)
    return 0.5 * torch.randn(int(view[NODE_POS_KEY].shape[0]), 3,
                             generator=generator)


def _packet_from_view(
        view: dict, *, seed: int = SEED,
        u0_nodes: torch.Tensor | None = None) -> CoarsePacket:
    """A synthetic encoder output whose SHAPES match ``view`` exactly.

    ``u0`` is drawn key-ordered (as the real decoder emits it, §5.8: "不得重排
    ``edge_index`` 或改变 ``key_index`` 行序") and then REPLACED by
    ``u0_nodes[key_index]`` when per-node values are wanted.  The draw order is
    kept identical in both branches so that ``z``/``node_h``/``q0_raw``/``e0``
    are bit-identical across the two constructions.
    """
    generator = torch.Generator().manual_seed(seed)
    n_nodes = int(view[NODE_POS_KEY].shape[0])
    n_edges = int(view[EDGE_INDEX_KEY].shape[1])
    n_keys = int(view[NODE_KEY_SET_KEY].shape[0])
    n_graphs = int(view[GRAPH_CV_DELTA_KEY].shape[0])
    z = torch.randn(n_graphs, ENCODER_Z_DIM_V543, generator=generator)
    node_h = torch.randn(n_nodes, KEY_S_DIM_V543, generator=generator)
    drawn_u0 = 0.5 * torch.randn(n_keys, 3, generator=generator)
    q0_raw = torch.randn(n_graphs, generator=generator)
    e0 = torch.randn(n_graphs, generator=generator)
    if u0_nodes is not None:
        drawn_u0 = u0_nodes[view[NODE_KEY_SET_KEY]]
    return CoarsePacket(
        z=z,
        edge_msg=torch.zeros(n_edges, 48),
        node_h=node_h,
        u0=drawn_u0,
        q0_raw=q0_raw,
        e0=e0,
    )


def _rotation(seed: int) -> torch.Tensor:
    """A deterministic proper rotation ``[3,3]`` (QR of a seeded Gaussian)."""
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(matrix)
    q = q * torch.sign(torch.diagonal(r))
    if float(torch.linalg.det(q)) < 0.0:
        q[:, 0] = -q[:, 0]
    return q.to(torch.float32)


def _rotate_view(view: dict, rotation: torch.Tensor) -> dict:
    out = dict(view)
    out[NODE_POS_KEY] = torch.matmul(view[NODE_POS_KEY], rotation.t())
    return out


def _translate_view(view: dict, shift: torch.Tensor) -> dict:
    out = dict(view)
    out[NODE_POS_KEY] = view[NODE_POS_KEY] + shift
    return out


def _slot_nodes(bridge: LatentBridge, view: dict) -> list[int]:
    """key slot → global node row (from the reconstructed pair set)."""
    return [int(row) for row in
            bridge.reconstruct_changed_pairs(view).key_index.tolist()]


def _mirror_pair_row(row: torch.Tensor) -> torch.Tensor:
    """The same unordered pair seen with the OPPOSITE canonical orientation.

    ``pair_s`` is canonical in key-slot order (§5.4 :379), so re-indexing the
    keys can flip which endpoint is ``i``.  Documented covariance of the 48
    dims under that flip:

    * 0–3 (``b_src``/``b_tgt``/``ΔB``/``|ΔB|``) unchanged — they are carried
      from the forward contract, which has no atom-level arrow (F2 :139);
    * 22 (``ΔZ/10``) negates and 40–47 (``W_fixed @ (key_s[i]-key_s[j])``)
      negate, because both are oriented differences;
    * 24–27 and 28–31 swap (the two endpoint-role blocks);
    * everything else is orientation free.
    """
    mirrored = row.clone()
    start_i, stop_i = PAIR_S_LAYOUT["role_i"]
    start_j, stop_j = PAIR_S_LAYOUT["role_j"]
    mirrored[start_i:stop_i] = row[start_j:stop_j]
    mirrored[start_j:stop_j] = row[start_i:stop_i]
    mirrored[22] = -row[22]
    start, stop = PAIR_S_LAYOUT["sketch"]
    mirrored[start:stop] = -row[start:stop]
    return mirrored


def _tiny_view() -> dict:
    """A HAND-COMPUTABLE single-graph system (5 atoms, 3 keys, 1 changed pair).

    Geometry::

        atom 0 (key)  (0,0,0)  atom 1 (key) (2,0,0)  atom 2 (0,3,0)
        atom 3        (0,0,4)  atom 4 (key) (10,0,0)  <-- isolated key

    Forward edges (each stored both directions, exactly like
    ``build_sample_features_v3``)::

        (0,1) B_src=1.0 B_tgt=0.5  covalent -> ΔB=-0.5  (breaking)
        (0,2) B_src=1.0 B_tgt=1.0  covalent -> ΔB= 0    (unchanged)
        (1,2) B_src=0.0 B_tgt=1.0  covalent -> ΔB=+1.0  (forming)
        (0,3)                       spatial -> ΔB= 0

    So ``K = 3``, ``P = 1`` (pair (0,1)), and key slot 2 (atom 4) has no
    incident edge at all — the zero-denominator path of §5.3 :372.
    """
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0],
        [10.0, 0.0, 0.0],
    ], dtype=torch.float32)
    numbers = torch.tensor([6, 8, 6, 1, 6], dtype=torch.int64)
    key_index = torch.tensor([0, 1, 4], dtype=torch.int64)
    graph_index = torch.zeros(5, dtype=torch.int64)
    intent_role = torch.tensor([
        [1.0, 0.0, 0.0],   # 0: breaking
        [1.0, 1.0, 0.0],   # 1: breaking + forming
        [0.0, 1.0, 0.0],   # 2: forming
        [0.0, 0.0, 1.0],   # 3: spectator
        [0.0, 0.0, 1.0],   # 4: spectator
    ], dtype=torch.float32)

    # (a, b, B_src, B_tgt, covalent, aromatic, spatial)
    undirected = (
        (0, 1, 1.0, 0.5, 1.0, 0.0, 0.0),
        (0, 2, 1.0, 1.0, 1.0, 0.0, 0.0),
        (1, 2, 0.0, 1.0, 1.0, 0.0, 0.0),
        (0, 3, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    src: list[int] = []
    dst: list[int] = []
    b_src: list[float] = []
    b_tgt: list[float] = []
    attrs: list[list[float]] = []
    for a, b, order_src, order_tgt, covalent, aromatic, spatial in undirected:
        row = [0.0] * 17
        row[EDGE_PACK_SLOTS_V3["is_spatial"]] = spatial
        row[EDGE_PACK_SLOTS_V3["covalent_union"]] = covalent
        row[EDGE_PACK_SLOTS_V3["aromatic_union"]] = aromatic
        for first, second in ((a, b), (b, a)):
            src.append(first)
            dst.append(second)
            b_src.append(order_src)
            b_tgt.append(order_tgt)
            attrs.append(row)

    delta = torch.tensor(b_tgt, dtype=torch.float32) - torch.tensor(
        b_src, dtype=torch.float32)
    return {
        NODE_Z_KEY: numbers,
        NODE_POS_KEY: pos,
        NODE_GRAPH_KEY: graph_index,
        NODE_KEY_SET_KEY: key_index,
        NODE_INTENT_ROLE_KEY: intent_role,
        EDGE_INDEX_KEY: torch.tensor([src, dst], dtype=torch.int64),
        EDGE_ATTR_KEY: torch.tensor(attrs, dtype=torch.float32),
        EDGE_B_SRC_KEY: torch.tensor(b_src, dtype=torch.float32),
        EDGE_B_TGT_KEY: torch.tensor(b_tgt, dtype=torch.float32),
        EDGE_DELTA_B_KEY: delta,
        EDGE_ABS_DELTA_KEY: delta.abs(),
        EDGE_IS_CHANGED_KEY: (delta.abs() > 0).to(torch.float32),
        GRAPH_CV_DELTA_KEY: torch.tensor(
            [[-1.0, 0.5, 0.0, 0.0]], dtype=torch.float32),
        GRAPH_SIGNED_INTENT_KEY: torch.tensor([-1.0], dtype=torch.float32),
    }


def _unit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``(a - b)/(‖a-b‖ + 0.1)`` — the plan §5.3 ``unit_e`` convention."""
    difference = a - b
    return difference / (difference.norm() + 0.1)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bridge() -> LatentBridge:
    return LatentBridge()


@pytest.fixture(scope="module")
def fixture_view() -> dict:
    return forward_view_v3(make_v3_gate_batch(n_graphs=2, jitter_seed=SEED))


@pytest.fixture(scope="module")
def fixture_context(bridge: LatentBridge, fixture_view: dict) -> DecoderContextV1:
    return bridge.build_context(_packet_from_view(fixture_view), fixture_view)


@pytest.fixture(scope="module")
def tiny_view() -> dict:
    return _tiny_view()


@pytest.fixture(scope="module")
def tiny_context(bridge: LatentBridge, tiny_view: dict) -> DecoderContextV1:
    return bridge.build_context(_packet_from_view(tiny_view), tiny_view)


# ---------------------------------------------------------------------------
# 1. frozen constants and the field table
# ---------------------------------------------------------------------------
def test_bridge_constants_match_plan() -> None:
    assert A_REF_V543 == 0.08                       # plan §5.3 :357
    assert KEY_V_CHANNELS_V543 == 8                 # plan §5.2/§5.3
    assert PAIR_S_DIM_V543 == 48                    # plan §5.2/§5.4
    assert KEY_EDGE_ATTR_DIM_V543 == 8              # plan §5.5 :403
    assert GRAPH_S_DIM_V543 == ENCODER_Z_DIM_V543 == 64
    assert KEY_S_DIM_V543 == 48
    assert DECODER_CONTEXT_SCHEMA_V543 == "decoder_context_v1"
    assert PRECISION_VERSION_V543 == "fp32_v1"


def test_pair_s_layout_covers_48_contiguously() -> None:
    covered: list[int] = []
    for _name, (start, stop) in PAIR_S_LAYOUT.items():
        assert stop > start
        covered.extend(range(start, stop))
    assert covered == list(range(PAIR_S_DIM_V543))


def test_context_field_table_is_the_plan_table() -> None:
    names = tuple(field.name for field in dataclasses.fields(DecoderContextV1))
    assert names == DECODER_CONTEXT_FIELDS
    assert len(names) == 22
    assert names[-4:] == DECODER_CONTEXT_IDENTITY_FIELDS
    assert set(DECODER_CONTEXT_TENSOR_FIELDS) == set(names) - set(
        DECODER_CONTEXT_IDENTITY_FIELDS)
    # §5.6 :427 "``__slots__`` 只含 §5.2 字段"
    assert set(DecoderContextV1.__slots__) == set(DECODER_CONTEXT_FIELDS)
    assert DecoderContextV1.__dataclass_params__.frozen is True


def test_bridge_is_parameter_free(bridge: LatentBridge) -> None:
    assert list(bridge.parameters()) == []
    assert int(sum(p.numel() for p in bridge.parameters())) == 0
    assert "w_fixed" not in bridge.state_dict()      # persistent=False (§5.1)
    assert tuple(bridge.w_fixed.shape) == (8, KEY_S_DIM_V543)
    scale = 1.0 / math.sqrt(KEY_S_DIM_V543)
    assert torch.allclose(bridge.w_fixed.abs(),
                          torch.full_like(bridge.w_fixed, scale))
    row_norms = bridge.w_fixed.norm(dim=1)
    assert torch.allclose(row_norms, torch.ones(8), atol=1e-6)
    spec = bridge.describe()
    assert spec["parameters"] == 0
    assert spec["bridge_spec_sha256"] == BRIDGE_SPEC_SHA256
    assert len(BRIDGE_SPEC_SHA256) == 64
    assert bridge_spec_sha256() == BRIDGE_SPEC_SHA256


# ---------------------------------------------------------------------------
# 2. pair set reconstruction (F2) + ownership of the supervision keys
# ---------------------------------------------------------------------------
def test_v543_pair_index_ownership(bridge: LatentBridge) -> None:
    """F2 / §5.6 :431–435 — read-only comparison against ``pair_index``."""
    batch = make_v3_batch(n_graphs=2, jitter_seed=SEED)
    view = forward_view_v3(batch)
    supervised = batch["pair_index"]                  # supervision, test only
    record = audit_pair_sets(bridge, view, supervised_pair_index=supervised)
    assert record["schema"] == PAIR_AUDIT_SCHEMA
    assert record["n_bridge_pairs"] == record["n_supervised_pairs"]
    assert record["n_bridge_pairs"] == int(batch["n_pairs"].sum())
    assert record["n_missing_from_bridge"] == 0
    assert record["n_extra_in_bridge"] == 0
    assert record["endpoints_match"] is True
    assert record["supervision_read"] == "compare_only"

    # |ΔB| agrees pair for pair — but ONLY after the frozen feature-contract
    # rescaling: the forward view's ``b_src``/``b_tgt``/``delta_b_intent`` are
    # stored as bond_order / BOND_SCALE (= 3.0, key_atom_graph_v3.py:182–183),
    # while the supervision ``pair_edit`` is the RAW bond-order edit
    # (key_atom_graph_v3.py:245).  Asserting the exact factor-3 relation keeps
    # the check strong and pins the discrepancy instead of hiding it behind a
    # tolerance.  (REPORTED to the orchestrator: this is a data-contract
    # mismatch, not a Bridge behaviour.)
    pairs = bridge.reconstruct_changed_pairs(view)
    ours = {
        (int(min(a, b)), int(max(a, b))): float(value)
        for (a, b), value in zip(pairs.pair_node_index.t().tolist(),
                                 pairs.abs_delta.tolist())
    }
    theirs = {
        (int(min(a, b)), int(max(a, b))): abs(float(value))
        for (a, b), value in zip(supervised.t().tolist(),
                                 batch["pair_edit"].tolist())
    }
    assert set(ours) == set(theirs)
    for key, value in ours.items():
        assert abs(value * BOND_SCALE - theirs[key]) <= 1e-5, key
        assert abs(value) <= 1.0 + 1e-6, key     # forward-side normalization


def test_supervision_never_enters_the_bridge(bridge: LatentBridge) -> None:
    """The structural separation: the model path cannot receive supervision."""
    batch = make_v3_batch(n_graphs=2, jitter_seed=SEED)   # carries pair_index
    view = forward_view_v3(batch)                         # which is stripped
    packet = _packet_from_view(view)
    # the fixture carries the CORE supervision keys (the id tuples are added by
    # other collate paths, so assert presence per key below instead of assuming
    # the whole tuple is present in one fixture)
    present = sorted(set(SUPERVISION_KEYS_V3) & set(batch))
    assert {"y_teacher", "pair_index", "pair_edit", "frame_weight",
            "d0"} <= set(present)
    with pytest.raises(V543SupervisionLeakError) as caught:
        bridge.build_context(packet, batch)
    assert caught.value.code == "V543_SUPERVISION_LEAK"
    # EVERY supervision key is refused on its own, present in the fixture or not
    for name in SUPERVISION_KEYS_V3:
        poisoned = dict(view)
        poisoned[name] = torch.zeros(1)
        with pytest.raises(V543SupervisionLeakError) as single:
            bridge.reconstruct_changed_pairs(poisoned)
        assert single.value.code == "V543_SUPERVISION_LEAK"
        assert name in single.value.detail
    # the forward view alone is accepted
    context = bridge.build_context(packet, view)
    assert isinstance(context, DecoderContextV1)


@pytest.mark.parametrize("name", SUPERVISION_KEYS_V3)
def test_each_supervision_key_is_a_typed_refusal(
        bridge: LatentBridge, name: str) -> None:
    view = dict(_tiny_view())
    view[name] = torch.tensor([12345.0])
    with pytest.raises(V543SupervisionLeakError) as caught:
        bridge.reconstruct_changed_pairs(view)
    assert caught.value.code == "V543_SUPERVISION_LEAK"
    assert name in caught.value.detail


def test_non_forward_key_is_refused(bridge: LatentBridge) -> None:
    view = dict(_tiny_view())
    view["ptr"] = torch.tensor([0])
    with pytest.raises(V543BridgeError) as caught:
        bridge.reconstruct_changed_pairs(view)
    assert caught.value.code == "V543_BRIDGE_FORWARD_VIEW_LEAK"
    assert set(_tiny_view()) <= set(FORWARD_VIEW_KEYS_V3)


def test_pairs_are_canonical_with_signed_delta(bridge: LatentBridge) -> None:
    pairs = bridge.reconstruct_changed_pairs(_tiny_view())
    assert pairs.n_pairs == 1
    assert pairs.pair_key_slots.tolist() == [[0], [1]]      # i < j (key slots)
    assert pairs.pair_node_index.tolist() == [[0], [1]]      # global node rows
    assert float(pairs.delta_b[0]) == pytest.approx(-0.5)    # 0.5 - 1.0
    assert float(pairs.abs_delta[0]) == pytest.approx(0.5)
    assert float(pairs.is_covalent[0]) == 1.0
    assert float(pairs.is_spatial[0]) == 0.0
    # Counting convention: ``stats`` counts EDGES, and the frozen builder stores
    # every undirected pair in BOTH directions (§5.5 "双向存储").  This fixture
    # has TWO changed undirected pairs — (0,1) with ΔB=-0.5 and (1,2) with
    # ΔB=+1.0 — i.e. 4 changed directed edges; only (0,1) has BOTH endpoints in
    # the K set (key_index = [0,1,4], so atom 2 is not a key), so it is the one
    # K-pair and the other 2 changed edges are counted as non-key incidents.
    assert pairs.stats["n_changed_edges"] == 4
    assert pairs.stats["n_pair_edges"] == 2
    assert pairs.stats["n_nonkey_changed_edges"] == 2


def test_edge_order_does_not_change_the_context(
        bridge: LatentBridge, tiny_view: dict) -> None:
    packet = _packet_from_view(tiny_view)
    reference = bridge.build_context(packet, tiny_view)
    permutation = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.int64)
    shuffled = dict(tiny_view)
    shuffled[EDGE_INDEX_KEY] = tiny_view[EDGE_INDEX_KEY][:, permutation]
    for name in (EDGE_ATTR_KEY, EDGE_B_SRC_KEY, EDGE_B_TGT_KEY,
                 EDGE_DELTA_B_KEY, EDGE_ABS_DELTA_KEY, EDGE_IS_CHANGED_KEY):
        shuffled[name] = tiny_view[name][permutation]
    rebuilt = bridge.build_context(packet, shuffled)
    for name in DECODER_CONTEXT_TENSOR_FIELDS:
        left = getattr(rebuilt, name)
        right = getattr(reference, name)
        if left.is_floating_point():
            # the edge order changes the fp32 accumulation order of the
            # channel sums, so equality is within fp32 rounding, not bitwise
            assert torch.allclose(left, right, atol=1e-6), name
        else:
            assert torch.equal(left, right), name


def test_inconsistent_delta_is_refused(bridge: LatentBridge) -> None:
    view = dict(_tiny_view())
    delta = view[EDGE_DELTA_B_KEY].clone()
    # the (0,1) pair appears twice (both directions): flip ONE of them, which
    # destroys the sign reference the forward contract provides (§5.6 audit).
    delta[1] = 0.5
    view[EDGE_DELTA_B_KEY] = delta
    with pytest.raises(V543BridgeError) as caught:
        bridge.reconstruct_changed_pairs(view)
    assert caught.value.code == "V543_BRIDGE_PAIR_DELTA_CONFLICT"


# ---------------------------------------------------------------------------
# 3. symmetries (§5.2 table columns: 旋转 / 平移 / 置换)
# ---------------------------------------------------------------------------
def test_v543_rotation_equivariance(
        bridge: LatentBridge, fixture_view: dict) -> None:
    """Rotating the input rotates ``key_v``/``u0``; everything else is invariant.

    ``u0``'s contract source is the ENCODER (``DirectMoveDecoder``, plan §5.2
    来源 column) and it is a POLAR vector, so an E(3)-equivariant encoder emits
    the rotated ``u0`` for the rotated geometry.  Both builds therefore use the
    matching encoder output for their own geometry — otherwise the test would
    wrongly demand that the Bridge rotate an encoder tensor it only forwards
    (§5.1 principle 1: the Bridge is parameter free and reads ``pos`` only as a
    difference, §5.6 :411).  ``key_v`` channel 0 is ``u0/a_ref`` (§5.3 :361),
    so it rotates with ``u0``; no assertion is weakened.
    """
    rotation = _rotation(SEED + 1)
    rotated_view = _rotate_view(fixture_view, rotation)
    u0_nodes = _node_u0(fixture_view)
    reference = bridge.build_context(
        _packet_from_view(fixture_view, u0_nodes=u0_nodes), fixture_view)
    rotated = bridge.build_context(
        _packet_from_view(rotated_view,
                          u0_nodes=torch.matmul(u0_nodes, rotation.t())),
        rotated_view)

    # polar vectors rotate; invariant fields do not move (§5.2 table)
    assert torch.allclose(rotated.key_v,
                          torch.matmul(reference.key_v, rotation.t()),
                          atol=1e-5)
    assert torch.allclose(rotated.u0,
                          torch.matmul(reference.u0, rotation.t()),
                          atol=1e-5)
    for name in ("graph_s", "key_s", "q0_raw", "endpoint0_logit"):
        assert torch.equal(getattr(rotated, name),
                           getattr(reference, name)), name
    for name in ("pair_s", "key_edge_attr"):
        assert torch.allclose(getattr(rotated, name),
                              getattr(reference, name), atol=1e-4), name
    for name in ("key_graph", "key_ptr", "pair_ptr", "token_ptr",
                 "token_index", "token_type", "token_graph",
                 "intent_pair_index", "pair_graph", "key_edge_index"):
        assert torch.equal(getattr(rotated, name),
                           getattr(reference, name)), name


def test_v543_translation_invariance(
        bridge: LatentBridge, fixture_view: dict) -> None:
    """A global shift changes nothing — not even ``u0`` (a displacement, §5.2)."""
    shift = torch.tensor([7.5, -3.25, 11.0], dtype=torch.float32)
    u0_nodes = _node_u0(fixture_view)
    reference = bridge.build_context(
        _packet_from_view(fixture_view, u0_nodes=u0_nodes), fixture_view)
    moved = bridge.build_context(
        _packet_from_view(_translate_view(fixture_view, shift),
                          u0_nodes=u0_nodes),
        _translate_view(fixture_view, shift))
    for name in DECODER_CONTEXT_TENSOR_FIELDS:
        left = getattr(moved, name)
        right = getattr(reference, name)
        if left.is_floating_point():
            assert torch.allclose(left, right, atol=1e-5), name
        else:
            assert torch.equal(left, right), name


def test_v543_permutation_equivariance(
        bridge: LatentBridge, fixture_view: dict) -> None:
    """Reversing the key order permutes rows with the documented covariance.

    ``u0`` is supplied PER NODE (encoder output, key-order dependent only by
    row indexing), so a key re-indexing must be a pure row permutation of
    ``key_s``/``u0``/``key_v`` — that is the exact claim under test.
    """
    view = dict(fixture_view)
    view[NODE_KEY_SET_KEY] = fixture_view[NODE_KEY_SET_KEY].flip(0)
    u0_nodes = _node_u0(fixture_view)
    reference = bridge.build_context(
        _packet_from_view(fixture_view, u0_nodes=u0_nodes), fixture_view)
    permuted = bridge.build_context(
        _packet_from_view(view, u0_nodes=u0_nodes), view)

    old_nodes = _slot_nodes(bridge, fixture_view)
    new_nodes = _slot_nodes(bridge, view)
    new_slot = {node: slot for slot, node in enumerate(new_nodes)}
    old_slot = {node: slot for slot, node in enumerate(old_nodes)}
    assert sorted(new_nodes) == sorted(old_nodes)

    for node in old_nodes:
        i, j = old_slot[node], new_slot[node]
        assert torch.equal(permuted.key_s[j], reference.key_s[i])
        assert torch.allclose(permuted.key_v[j], reference.key_v[i],
                              atol=1e-6)
        assert torch.equal(permuted.u0[j], reference.u0[i])

    # pair rows: the canonical orientation may flip, which swaps the two
    # endpoint-role blocks (F2: the direction is the canonical key-slot order).
    def rows_by_nodes(context, nodes):
        out = {}
        lo, hi = context.intent_pair_index.tolist()
        for row, (a, b) in enumerate(zip(lo, hi)):
            out[frozenset((nodes[a], nodes[b]))] = context.pair_s[row]
        return out

    old_rows = rows_by_nodes(reference, old_nodes)
    new_rows = rows_by_nodes(permuted, new_nodes)
    assert set(old_rows) == set(new_rows)
    for key, row in new_rows.items():
        expected = old_rows[key]
        assert (torch.allclose(row, expected, atol=1e-6)
                or torch.allclose(row, _mirror_pair_row(expected),
                                  atol=1e-6)), key

    # key-edge attributes are functions of the pair alone → unchanged per pair
    def edge_rows(context, nodes):
        out = {}
        for column in range(int(context.key_edge_index.shape[1])):
            a = nodes[int(context.key_edge_index[0, column])]
            b = nodes[int(context.key_edge_index[1, column])]
            out[(a, b)] = context.key_edge_attr[column]
        return out

    old_edges = edge_rows(reference, old_nodes)
    new_edges = edge_rows(permuted, new_nodes)
    assert set(old_edges) == set(new_edges)
    for key, row in new_edges.items():
        assert torch.allclose(row, old_edges[key], atol=1e-6), key


# ---------------------------------------------------------------------------
# 4. key_v — hand computed, page by page (§5.3)
# ---------------------------------------------------------------------------
def test_key_v_channels_hand_computed(tiny_view: dict, tiny_context) -> None:
    pos = tiny_view[NODE_POS_KEY]
    x0, x1, x2, x3, x4 = (pos[i] for i in range(5))
    key_v = tiny_context.key_v
    assert tuple(key_v.shape) == (3, KEY_V_CHANNELS_V543, 3)   # K=3 keys

    d01 = float((x1 - x0).norm())
    d02 = float((x2 - x0).norm())
    d03 = float((x3 - x0).norm())
    d12 = float((x2 - x1).norm())

    # ch1/ch2/ch3/ch6 are directed-edge sums with dst = the key atom; the
    # "distance-weighted" ch6 divides by Σ(d/4) (plan §5.3 :367).
    weights0 = (d01 / 4.0, d02 / 4.0, d03 / 4.0)
    key0_ch6 = (weights0[0] * _unit(x1, x0)
                + weights0[1] * _unit(x2, x0)
                + weights0[2] * _unit(x3, x0)) / sum(weights0)
    weights1 = (d01 / 4.0, d12 / 4.0)
    key1_ch6 = (weights1[0] * _unit(x0, x1)
                + weights1[1] * _unit(x2, x1)) / sum(weights1)

    expected = {
        # ch0 = u0 / a_ref
        "u0_over_a_ref": tiny_context.u0 / A_REF_V543,
        "breaking": torch.stack([_unit(x1, x0), _unit(x0, x1),
                                 torch.zeros(3)]),
        "forming": torch.stack([torch.zeros(3), _unit(x2, x1),
                                torch.zeros(3)]),
        "unchanged_covalent": torch.stack([_unit(x2, x0), torch.zeros(3),
                                           torch.zeros(3)]),
        "spatial_kappa": torch.stack([_unit(x3, x0), torch.zeros(3),
                                      torch.zeros(3)]),
        "critical_pair": torch.stack([
            -1.0 * _unit(x1, x0),     # sign(ΔB=-0.5) = -1
            -1.0 * _unit(x0, x1),
            torch.zeros(3)]),
        "distance_weighted": torch.stack([key0_ch6, key1_ch6,
                                          torch.zeros(3)]),
        "key_centroid": torch.stack([
            # mean of the OTHER keys' positions, minus this key (§5.3 :368)
            _unit((x1 + x4) / 2.0, x0),
            _unit((x0 + x4) / 2.0, x1),
            _unit((x0 + x1) / 2.0, x4),
        ]),
    }
    assert KEY_V_CHANNEL_NAMES == tuple(expected)
    for channel, name in enumerate(KEY_V_CHANNEL_NAMES):
        assert torch.allclose(key_v[:, channel], expected[name],
                              atol=1e-6), name


def test_key_v_zero_denominator_is_a_zero_vector(tiny_view: dict,
                                                 tiny_context) -> None:
    """§5.3 :372 — an isolated key has zero (not NaN) channels 1..6."""
    isolated = tiny_context.key_v[2]
    assert bool(torch.isfinite(isolated).all())
    for channel in range(1, 7):
        assert torch.equal(isolated[channel], torch.zeros(3)), channel
    # ch0 and ch7 stay defined
    assert torch.allclose(isolated[0], tiny_context.u0[2] / A_REF_V543)


def test_key_v_survives_single_key_graphs(bridge: LatentBridge) -> None:
    """ch7 has no "other key" in a 1-key graph → zero, never NaN."""
    view = dict(_tiny_view())
    view[NODE_KEY_SET_KEY] = torch.tensor([0], dtype=torch.int64)
    view[EDGE_IS_CHANGED_KEY] = torch.zeros_like(view[EDGE_IS_CHANGED_KEY])
    context = bridge.build_context(_packet_from_view(view), view)
    assert tuple(context.key_v.shape) == (1, 8, 3)
    assert bool(torch.isfinite(context.key_v).all())
    assert torch.equal(context.key_v[0, 7], torch.zeros(3))


def test_key_v_rotation_is_exactly_the_channel_symmetry(
        bridge: LatentBridge, tiny_view: dict) -> None:
    """Every one of the 8 channels is a polar vector and rotates with the input.

    ``u0`` (channel 0's source) comes from the ENCODER (plan §5.2 来源 column),
    so the rotated build gets the matching rotated encoder output; the other
    seven channels are built from ``pos`` differences only and must rotate by
    construction.
    """
    rotation = _rotation(SEED + 5)
    u0_nodes = _node_u0(tiny_view)
    reference = bridge.build_context(
        _packet_from_view(tiny_view, u0_nodes=u0_nodes), tiny_view)
    rotated = bridge.build_context(
        _packet_from_view(_rotate_view(tiny_view, rotation),
                          u0_nodes=torch.matmul(u0_nodes, rotation.t())),
        _rotate_view(tiny_view, rotation))
    for channel in range(KEY_V_CHANNELS_V543):
        assert torch.allclose(
            rotated.key_v[:, channel],
            torch.matmul(reference.key_v[:, channel], rotation.t()),
            atol=1e-5), channel


# ---------------------------------------------------------------------------
# 5. pair_s — 48 dims, segment by segment (§5.4)
# ---------------------------------------------------------------------------
def test_pair_s_hand_computed(tiny_view: dict, tiny_context) -> None:
    pair_s = tiny_context.pair_s
    assert tuple(pair_s.shape) == (1, PAIR_S_DIM_V543)
    row = pair_s[0]
    pos = tiny_view[NODE_POS_KEY]
    distance = float((pos[0] - pos[1]).norm())      # atoms 0 and 1 → d = 2.0
    assert distance == pytest.approx(2.0)

    # 0–5: b_src, b_tgt, ΔB, |ΔB|, is_changed, is_covalent
    assert float(row[0]) == pytest.approx(1.0)
    assert float(row[1]) == pytest.approx(0.5)
    assert float(row[2]) == pytest.approx(-0.5)
    assert float(row[3]) == pytest.approx(0.5)
    assert float(row[4]) == 1.0
    assert float(row[5]) == 1.0
    # 6–7: is_aromatic, is_spatial
    assert float(row[6]) == 0.0
    assert float(row[7]) == 0.0

    # 8–19: RBF12 with the frozen gaussian_rbf_v3 centres (σ = 0.3)
    centers = torch.linspace(0.0, 4.5, 12, dtype=torch.float32)
    expected_rbf = torch.exp(-0.5 * ((distance - centers) / 0.3) ** 2)
    assert torch.allclose(row[8:20], expected_rbf, atol=1e-5)

    # 20–23: d, 1/(d+1), ΔZ/10, |ΔZ|/10  (z0 = 6, z1 = 8)
    assert float(row[20]) == pytest.approx(distance)
    assert float(row[21]) == pytest.approx(1.0 / (distance + 1.0))
    assert float(row[22]) == pytest.approx((6.0 - 8.0) / 10.0)
    assert float(row[23]) == pytest.approx(0.2)

    # 24–31: endpoint roles + is_key, endpoint i = key slot 0 (atom 0)
    assert row[24:28].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert row[28:32].tolist() == [1.0, 1.0, 0.0, 1.0]

    # 32–39: graph broadcast = signed_intent, ‖cv_delta‖, cv_delta[0:6] padded
    assert float(row[32]) == pytest.approx(-1.0)
    assert float(row[33]) == pytest.approx(math.sqrt(1.0 + 0.25))
    assert row[34:38].tolist() == pytest.approx([-1.0, 0.5, 0.0, 0.0])
    assert row[38:40].tolist() == [0.0, 0.0]

    # 40–47: W_fixed @ (key_s[i] - key_s[j]) — the frozen sketch
    key_s = tiny_context.key_s
    expected_sketch = torch.matmul(key_s[0] - key_s[1],
                                   fixed_sketch_matrix().t())
    assert torch.allclose(row[40:48], expected_sketch, atol=1e-5)


def test_pair_s_is_rotation_and_translation_invariant(
        bridge: LatentBridge, tiny_view: dict) -> None:
    reference = bridge.build_context(_packet_from_view(tiny_view), tiny_view)
    rotated = bridge.build_context(
        _packet_from_view(tiny_view),
        _rotate_view(tiny_view, _rotation(SEED + 9)))
    moved = bridge.build_context(
        _packet_from_view(tiny_view),
        _translate_view(tiny_view, torch.tensor([3.0, -1.0, 2.5])))
    assert torch.allclose(rotated.pair_s, reference.pair_s, atol=1e-4)
    assert torch.allclose(moved.pair_s, reference.pair_s, atol=1e-4)


# ---------------------------------------------------------------------------
# 6. key graph (§5.5) — bound, dedup, bidirectional, attributes
# ---------------------------------------------------------------------------
def test_key_graph_edges_are_deduped_and_bidirectional(
        fixture_context) -> None:
    index = fixture_context.key_edge_index
    attributes = fixture_context.key_edge_attr
    n_keys = int(fixture_context.key_s.shape[0])
    n_edges = int(index.shape[1])
    assert n_edges % 2 == 0
    assert tuple(attributes.shape) == (n_edges, KEY_EDGE_ATTR_DIM_V543)
    assert int(index.min()) >= 0 and int(index.max()) < n_keys
    assert bool((index[0] != index[1]).all())          # no self loops

    directed = {(int(a), int(b)) for a, b in index.t().tolist()}
    for a, b in directed:
        assert (b, a) in directed                       # stored both ways
    undirected = {tuple(sorted(pair)) for pair in directed}
    assert len(undirected) == n_edges // 2

    # every changed pair is present as an edge, with edge_type 0
    lo, hi = fixture_context.intent_pair_index.tolist()
    changed_pairs = {tuple(sorted((int(a), int(b)))) for a, b in zip(lo, hi)}
    for pair in changed_pairs:
        assert pair in undirected
    for column, (a, b) in enumerate(index.t().tolist()):
        row = attributes[column]
        pair = tuple(sorted((int(a), int(b))))
        if pair in changed_pairs:
            assert float(row[6]) == 1.0                 # is_changed
            assert float(row[7]) == 0.0                 # edge_type covalent
        else:
            assert float(row[6]) == 0.0
            assert float(row[7]) == 1.0                 # edge_type spatial
            assert float(row[5]) == 0.0                 # |ΔB| = 0


def test_key_graph_attribute_dims(fixture_context) -> None:
    attributes = fixture_context.key_edge_attr
    index = fixture_context.key_edge_index
    assert attributes.shape[1] == 8
    assert bool((attributes[:, 0] >= 0.0).all())         # distance
    assert bool((attributes[:, 1:5] >= 0.0).all())       # 4 RBF channels
    assert bool((attributes[:, 1:5] <= 1.0 + 1e-6).all())
    assert bool((attributes[:, 5] >= 0.0).all())         # |ΔB|
    assert set(attributes[:, 7].tolist()) <= {0.0, 1.0}  # edge_type
    # the four RBF centres are 2,3,4,5 with σ = 0.3 (peak at the centre)
    centres = torch.tensor([2.0, 3.0, 4.0, 5.0])
    distance = attributes[:, 0]
    expected = torch.exp(-0.5 * ((distance.unsqueeze(-1)
                                  - centres.unsqueeze(0)) / 0.3) ** 2)
    assert torch.allclose(attributes[:, 1:5], expected, atol=1e-5)
    assert int(index.shape[1]) == int(attributes.shape[0])


def test_ek_bound_holds(fixture_context) -> None:
    n_keys = int(fixture_context.key_s.shape[0])
    n_pairs = int(fixture_context.pair_s.shape[0])
    n_edges = int(fixture_context.key_edge_attr.shape[0])
    assert n_edges <= 2 * (n_pairs + 4 * n_keys)


def test_knn_neighbours_respect_cutoff_and_count(
        bridge: LatentBridge, tiny_view: dict) -> None:
    """Only the changed pair survives: the keys are 2/8/10 Å apart (§5.5)."""
    context = bridge.build_context(_packet_from_view(tiny_view), tiny_view)
    index = context.key_edge_index
    pairs = {(int(a), int(b)) for a, b in index.t().tolist()}
    # key slots: 0 = atom 0, 1 = atom 1 (2.0 Å apart), 2 = atom 4 (isolated)
    assert pairs == {(0, 1), (1, 0)}
    assert int(index.shape[1]) == 2
    # the changed pair is flagged covalent/edge_type 0 on both directions
    for column in range(int(index.shape[1])):
        assert float(context.key_edge_attr[column, 6]) == 1.0   # is_changed
        assert float(context.key_edge_attr[column, 5]) == 0.5   # |ΔB|
        assert float(context.key_edge_attr[column, 7]) == 0.0   # edge_type
    # Ek ≤ 2·(P + 4K)
    assert int(index.shape[1]) <= 2 * (1 + 4 * 3)


# ---------------------------------------------------------------------------
# 7. token layout (§5.5)
# ---------------------------------------------------------------------------
def test_token_layout_per_graph(tiny_context) -> None:
    n_graphs = int(tiny_context.key_ptr.shape[0]) - 1
    assert n_graphs == 1
    assert tiny_context.key_ptr.tolist() == [0, 3]
    assert tiny_context.pair_ptr.tolist() == [0, 1]
    assert tiny_context.token_ptr.tolist() == [0, 5]
    # [graph, keys…, pairs…]
    assert tiny_context.token_index.tolist() == [-1, 0, 1, 2, 0]
    assert tiny_context.token_type.tolist() == [0, 1, 1, 1, 2]
    assert tiny_context.token_graph.tolist() == [0, 0, 0, 0, 0]
    assert tiny_context.token_type.dtype == torch.int8
    assert tiny_context.token_index.dtype == torch.int64
    assert tiny_context.token_graph.dtype == torch.int64


def test_token_layout_multi_graph(fixture_context) -> None:
    n_graphs = int(fixture_context.token_ptr.shape[0]) - 1
    n_keys = int(fixture_context.key_s.shape[0])
    n_pairs = int(fixture_context.pair_s.shape[0])
    assert int(fixture_context.token_ptr[-1]) == n_graphs + n_keys + n_pairs
    types = fixture_context.token_type.tolist()
    index = fixture_context.token_index.tolist()
    pointer = fixture_context.token_ptr.tolist()
    for graph in range(n_graphs):
        start, stop = pointer[graph], pointer[graph + 1]
        assert types[start] == 0 and index[start] == -1
        key_count = (int(fixture_context.key_ptr[graph + 1])
                     - int(fixture_context.key_ptr[graph]))
        assert types[start + 1:start + 1 + key_count] == [1] * key_count
        assert set(types[start + 1 + key_count:stop]) <= {2}
    # prefix sums are monotone and close (§5.2 :307–308)
    for name in ("key_ptr", "pair_ptr", "token_ptr"):
        pointer = getattr(fixture_context, name)
        assert int(pointer[0]) == 0
        assert bool((pointer[1:] >= pointer[:-1]).all())


# ---------------------------------------------------------------------------
# 8. empty sets: P = 0 and Ek = 0 (§5.2 :309–310)
# ---------------------------------------------------------------------------
def test_empty_pair_set_is_supported(bridge: LatentBridge) -> None:
    view = dict(_tiny_view())
    view[EDGE_DELTA_B_KEY] = torch.zeros_like(view[EDGE_DELTA_B_KEY])
    view[EDGE_ABS_DELTA_KEY] = torch.zeros_like(view[EDGE_ABS_DELTA_KEY])
    view[EDGE_IS_CHANGED_KEY] = torch.zeros_like(view[EDGE_IS_CHANGED_KEY])
    context = bridge.build_context(_packet_from_view(view), view)
    assert context.pair_s.shape == (0, PAIR_S_DIM_V543)
    assert context.intent_pair_index.shape == (2, 0)
    assert context.pair_ptr.tolist() == [0, 0]
    assert int(context.token_ptr[-1]) == 1 + 3 + 0        # G + K + P
    assert context.token_type.tolist() == [0, 1, 1, 1]
    validate_decoder_context(context)                     # still exact
    assert_no_supervision(context)


def test_empty_pair_set_with_spatial_key_edges(
        bridge: LatentBridge) -> None:
    """REGRESSION: ``P = 0`` AND ``Ek > 0`` at the same time (§5.2 :309).

    The spatial rows carry ``|ΔB| = 0`` / ``is_changed = 0`` / ``edge_type = 1``
    and come from the radius graph, not from the (empty) pair table — an
    unguarded ``pairs.abs_delta[...]`` gather over the zero-length pair table
    used to raise ``IndexError`` here.
    """
    view = dict(_tiny_view())
    view[EDGE_DELTA_B_KEY] = torch.zeros_like(view[EDGE_DELTA_B_KEY])
    view[EDGE_ABS_DELTA_KEY] = torch.zeros_like(view[EDGE_ABS_DELTA_KEY])
    view[EDGE_IS_CHANGED_KEY] = torch.zeros_like(view[EDGE_IS_CHANGED_KEY])
    # a second key 2.0 Å from key 0 so the radius graph is NON-empty while the
    # pair set is empty
    context = bridge.build_context(_packet_from_view(view), view)
    assert context.pair_s.shape == (0, PAIR_S_DIM_V543)
    assert int(context.pair_ptr[-1]) == 0
    n_edges = int(context.key_edge_index.shape[1])
    assert n_edges == 2                       # keys 0 and 1 are 2.0 Å apart
    attributes = context.key_edge_attr
    assert bool((attributes[:, 5] == 0.0).all())      # |ΔB| = 0 for every row
    assert bool((attributes[:, 6] == 0.0).all())      # is_changed = 0
    assert bool((attributes[:, 7] == 1.0).all())      # edge_type = spatial
    assert torch.allclose(attributes[:, 0],
                          torch.full((n_edges,),
                                     float((view[NODE_POS_KEY][1]
                                            - view[NODE_POS_KEY][0]).norm())))
    assert bool(torch.isfinite(attributes).all())
    validate_decoder_context(context)


def test_empty_key_edge_set_is_supported(bridge: LatentBridge) -> None:
    """K = 1, P = 0 → Ek = 0 (no message passing is required, §5.2 :310)."""
    view = dict(_tiny_view())
    view[NODE_KEY_SET_KEY] = torch.tensor([0], dtype=torch.int64)
    view[EDGE_IS_CHANGED_KEY] = torch.zeros_like(view[EDGE_IS_CHANGED_KEY])
    view[EDGE_DELTA_B_KEY] = torch.zeros_like(view[EDGE_DELTA_B_KEY])
    view[EDGE_ABS_DELTA_KEY] = torch.zeros_like(view[EDGE_ABS_DELTA_KEY])
    context = bridge.build_context(_packet_from_view(view), view)
    assert context.key_edge_index.shape == (2, 0)
    assert context.key_edge_attr.shape == (0, KEY_EDGE_ATTR_DIM_V543)
    assert context.key_edge_index.dtype == torch.int64
    assert context.key_edge_attr.dtype == torch.float32
    validate_decoder_context(context)


# ---------------------------------------------------------------------------
# 9. the two track views (§5.2.1 hard isolation)
# ---------------------------------------------------------------------------
def test_v543_track_view_isolation(fixture_context) -> None:
    move = fixture_context.move_view()
    process = fixture_context.process_view()
    assert isinstance(move, MoveContextV1)
    assert isinstance(process, ProcessContextV1)
    assert set(move.__slots__) == set(MOVE_CONTEXT_FIELDS)
    assert set(process.__slots__) == set(PROCESS_CONTEXT_FIELDS)
    assert context_field_names(move) == MOVE_CONTEXT_FIELDS
    assert context_field_names(process) == PROCESS_CONTEXT_FIELDS

    # §5.2.1 :346–349 — each view's OWN field set is disjoint from the fields
    # THAT view must refuse; the refused set is exactly the complement of the
    # view's field set inside the context's tensor fields, and every refused
    # field really is visible on the OTHER track.
    assert set(MOVE_CONTEXT_FIELDS).isdisjoint(MOVE_FORBIDDEN_FIELDS)
    assert set(PROCESS_CONTEXT_FIELDS).isdisjoint(PROCESS_FORBIDDEN_FIELDS)
    assert set(MOVE_FORBIDDEN_FIELDS) == (
        set(DECODER_CONTEXT_TENSOR_FIELDS) - set(MOVE_CONTEXT_FIELDS))
    assert set(PROCESS_FORBIDDEN_FIELDS) == (
        set(DECODER_CONTEXT_TENSOR_FIELDS) - set(PROCESS_CONTEXT_FIELDS))
    assert set(MOVE_FORBIDDEN_FIELDS) <= set(PROCESS_CONTEXT_FIELDS)
    assert set(PROCESS_FORBIDDEN_FIELDS) <= set(MOVE_CONTEXT_FIELDS)
    # together the two views cover the whole tensor surface exactly
    assert set(MOVE_CONTEXT_FIELDS) | set(PROCESS_CONTEXT_FIELDS) == set(
        DECODER_CONTEXT_TENSOR_FIELDS)


@pytest.mark.parametrize("name", MOVE_FORBIDDEN_FIELDS)
def test_move_view_refuses_process_fields(fixture_context, name: str) -> None:
    move = fixture_context.move_view()
    if name in MOVE_CONTEXT_FIELDS:      # pragma: no cover - disjoint by test
        pytest.skip(f"{name} is a MOVE field")
    with pytest.raises(V543TrackIsolationError) as caught:
        getattr(move, name)
    assert caught.value.code == "V543_TRACK_ISOLATION"
    assert isinstance(caught.value, AttributeError)
    assert name in caught.value.detail


@pytest.mark.parametrize("name", PROCESS_FORBIDDEN_FIELDS)
def test_process_view_refuses_move_fields(fixture_context, name: str) -> None:
    process = fixture_context.process_view()
    with pytest.raises(V543TrackIsolationError) as caught:
        getattr(process, name)
    assert caught.value.code == "V543_TRACK_ISOLATION"
    assert isinstance(caught.value, AttributeError)


@pytest.fixture(params=("move", "process", "context"))
def forbidden_target(request, fixture_context):
    """(label, object) pairs that must refuse every forbidden name."""
    if request.param == "move":
        return request.param, fixture_context.move_view()
    if request.param == "process":
        return request.param, fixture_context.process_view()
    return request.param, fixture_context


def test_v543_decoder_no_xyz(forbidden_target) -> None:
    """§5.6 :428 — any ``ctx.pos`` access must be an ``AttributeError``.

    Parametrised per surface so a regression names the side that broke (the two
    views refuse with the typed :class:`V543TrackIsolationError`, which IS an
    ``AttributeError``; the context itself has no ``__getattr__`` at all).
    """
    which, target = forbidden_target
    for name in ("pos", "xyz", "pair_index", "pair_edit", "y_teacher",
                 "q_true", "batch"):
        with pytest.raises(AttributeError):
            getattr(target, name)
    if which != "context":
        # and the refusal is the typed one, not a bare lookup failure
        with pytest.raises(V543TrackIsolationError) as caught:
            getattr(target, "pos")
        assert isinstance(caught.value, AttributeError)
        assert caught.value.code.startswith("V543_TRACK_")


def test_v543_context_carries_no_supervision_fields(fixture_context) -> None:
    assert_no_supervision(fixture_context)
    assert_no_supervision(fixture_context.move_view())
    assert_no_supervision(fixture_context.process_view())
    for name in SUPERVISION_KEYS_V3:
        assert name not in context_field_names(fixture_context)


def test_v543_no_supervision_leak(bridge: LatentBridge) -> None:
    """§5.6 :429 — sentinel supervision values never touch the context."""
    batch = make_v3_batch(n_graphs=2, jitter_seed=SEED)
    clean_view = forward_view_v3(batch)
    packet = _packet_from_view(clean_view)
    reference = bridge.build_context(packet, clean_view)

    poisoned = dict(clean_view)
    for name, sentinel in (("pair_edit", 12345.0), ("y_teacher", 999.0),
                           ("pair_index", 4321.0), ("frame_weight", 777.0)):
        poisoned[name] = torch.full((7,), sentinel)   # refused, never read
    with pytest.raises(V543SupervisionLeakError):
        bridge.build_context(packet, poisoned)

    for name in DECODER_CONTEXT_TENSOR_FIELDS:
        tensor = getattr(reference, name)
        if tensor.is_floating_point():
            assert not bool((tensor == 12345.0).any()), name
            assert not bool((tensor == 999.0).any()), name
            assert not bool((tensor == 4321.0).any()), name
            assert not bool((tensor == 777.0).any()), name
    # a second build from the clean view is bit-identical: nothing leaked in
    rebuilt = bridge.build_context(packet, clean_view)
    for name in DECODER_CONTEXT_TENSOR_FIELDS:
        assert torch.equal(getattr(rebuilt, name),
                           getattr(reference, name)), name
    assert_no_supervision(rebuilt)


def test_assert_no_supervision_catches_a_leak(fixture_context) -> None:
    @dataclasses.dataclass(frozen=True, slots=True)
    class Leaky:
        graph_s: torch.Tensor
        pair_index: torch.Tensor

    leaky = Leaky(graph_s=fixture_context.graph_s, pair_index=torch.zeros(1))
    with pytest.raises(V543SupervisionLeakError) as caught:
        assert_no_supervision(leaky)
    assert caught.value.code == "V543_CONTEXT_SUPERVISION"
    assert "pair_index" in caught.value.detail


# ---------------------------------------------------------------------------
# 10. integration with the encoder packet + validation
# ---------------------------------------------------------------------------
def test_context_matches_the_field_table(fixture_view: dict,
                                         fixture_context) -> None:
    n_graphs = int(fixture_view[GRAPH_CV_DELTA_KEY].shape[0])
    n_keys = int(fixture_view[NODE_KEY_SET_KEY].shape[0])
    n_nodes = int(fixture_view[NODE_POS_KEY].shape[0])
    n_edges = int(fixture_view[EDGE_INDEX_KEY].shape[1])
    assert fixture_context.schema == DECODER_CONTEXT_SCHEMA_V543
    assert fixture_context.precision_version == PRECISION_VERSION_V543
    assert len(fixture_context.bridge_spec_sha256) == 64
    assert fixture_context.encoder_ckpt_sha256 == ""

    assert fixture_context.graph_s.shape == (n_graphs, GRAPH_S_DIM_V543)
    assert fixture_context.key_s.shape == (n_keys, KEY_S_DIM_V543)
    assert fixture_context.key_v.shape == (n_keys, 8, 3)
    assert fixture_context.u0.shape == (n_keys, 3)
    assert fixture_context.key_graph.shape == (n_keys,)
    assert fixture_context.key_ptr.shape == (n_graphs + 1,)
    assert fixture_context.key_edge_attr.shape[1] == 8
    assert fixture_context.key_edge_attr.shape[0] == int(
        fixture_context.key_edge_index.shape[1])
    assert fixture_context.q0_raw.shape == (n_graphs,)
    assert fixture_context.endpoint0_logit.shape == (n_graphs,)

    for name in DECODER_CONTEXT_TENSOR_FIELDS:
        tensor = getattr(fixture_context, name)
        assert tensor.dtype in (
            torch.float32, torch.int64, torch.int8), name
        assert tensor.device.type == "cpu"
    # the packet really is the source of the graph representation
    packet = _packet_from_view(fixture_view)
    assert torch.equal(fixture_context.graph_s, packet.z)
    assert torch.equal(fixture_context.q0_raw, packet.q0_raw)
    assert torch.equal(fixture_context.endpoint0_logit, packet.e0)
    assert n_nodes > 0 and n_edges > 0
    validate_decoder_context(fixture_context)


def test_key_s_and_u0_come_from_the_packet(
        bridge: LatentBridge, fixture_view: dict) -> None:
    packet = _packet_from_view(fixture_view)
    context = bridge.build_context(packet, fixture_view)
    pairs = bridge.reconstruct_changed_pairs(fixture_view)
    assert torch.equal(context.key_s, packet.node_h[pairs.key_index])
    assert torch.equal(context.u0, packet.u0[pairs.key_permutation])
    assert torch.equal(context.key_graph, fixture_view[NODE_GRAPH_KEY][
        pairs.key_index])


def test_build_context_is_differentiable(
        bridge: LatentBridge, fixture_view: dict) -> None:
    """§5.7 :454 — ``joint_finetune`` must keep the autograd graph alive."""
    packet = _packet_from_view(fixture_view)
    node_h = packet.node_h.clone().requires_grad_(True)
    differentiable = CoarsePacket(
        z=packet.z, edge_msg=packet.edge_msg, node_h=node_h,
        u0=packet.u0, q0_raw=packet.q0_raw, e0=packet.e0)
    context = bridge.build_context(differentiable, fixture_view)
    assert bool(context.key_s.requires_grad)
    assert bool(context.pair_s.requires_grad)
    loss = context.pair_s.square().mean() + context.key_s.square().mean()
    loss.backward()
    assert node_h.grad is not None
    assert bool(torch.isfinite(node_h.grad).all())


def test_packet_mismatch_is_refused(bridge: LatentBridge,
                                    fixture_view: dict) -> None:
    packet = _packet_from_view(fixture_view)
    broken = CoarsePacket(
        z=packet.z, edge_msg=packet.edge_msg[:-1], node_h=packet.node_h,
        u0=packet.u0, q0_raw=packet.q0_raw, e0=packet.e0)
    with pytest.raises(V543BridgeError) as caught:
        bridge.build_context(broken, fixture_view)
    assert caught.value.code == "V543_BRIDGE_PACKET"


def test_validate_refuses_shape_and_dtype_drift(fixture_context) -> None:
    drifted = dataclasses.replace(
        fixture_context,
        key_v=fixture_context.key_v[:, :7, :])
    with pytest.raises(V543ContextError) as caught:
        validate_decoder_context(drifted)
    assert caught.value.code == "V543_CONTEXT_KEY_V_SHAPE"

    wrong_dtype = dataclasses.replace(
        fixture_context, pair_s=fixture_context.pair_s.to(torch.float64))
    with pytest.raises(V543ContextError) as caught:
        validate_decoder_context(wrong_dtype)
    assert caught.value.code == "V543_CONTEXT_DTYPE"

    wrong_schema = dataclasses.replace(fixture_context, schema="decoder_v0")
    with pytest.raises(V543ContextError) as caught:
        validate_decoder_context(wrong_schema)
    assert caught.value.code == "V543_CONTEXT_SCHEMA"


def test_build_stats_are_auditable(bridge: LatentBridge,
                                   fixture_view: dict) -> None:
    bridge.build_context(_packet_from_view(fixture_view), fixture_view)
    stats = bridge.last_build_stats
    pairs = bridge.reconstruct_changed_pairs(fixture_view)
    assert stats["n_pairs"] == pairs.n_pairs
    assert stats["n_keys"] == int(fixture_view[NODE_KEY_SET_KEY].numel())
    assert stats["n_key_edges"] == (stats["n_changed_key_edges"]
                                    + stats["n_spatial_key_edges"])
    assert stats["n_tokens"] == (int(fixture_view[GRAPH_CV_DELTA_KEY].shape[0])
                                 + stats["n_keys"] + stats["n_pairs"])
    assert stats["ek_bound"] == 2 * (pairs.n_pairs + 4 * stats["n_keys"])
    assert stats["bridge_spec_sha256"] == BRIDGE_SPEC_SHA256


# ---------------------------------------------------------------------------
# 11. cache identity (§5.7)
# ---------------------------------------------------------------------------
def test_cache_key_order_and_components() -> None:
    key = cache_key(
        encoder_ckpt_sha256="a" * 64,
        input_feature_sha256="b" * 64,
        source_frame_identity="c" * 64,
        code_fingerprint="d" * 12,
    )
    parts = key.split("|")
    assert len(parts) == len(CACHE_KEY_FIELDS) == 6
    assert parts[CACHE_KEY_FIELDS.index("encoder_ckpt_sha256")] == "a" * 64
    assert parts[CACHE_KEY_FIELDS.index("bridge_spec_sha256")] == \
        BRIDGE_SPEC_SHA256
    assert parts[CACHE_KEY_FIELDS.index("precision_version")] == \
        PRECISION_VERSION_V543
    # an incomplete identity must be refused, never silently reused (§5.7)
    with pytest.raises(V543BridgeError) as caught:
        cache_key(encoder_ckpt_sha256="a" * 64, input_feature_sha256="",
                  source_frame_identity="c" * 64, code_fingerprint="d" * 12)
    assert caught.value.code == "V543_BRIDGE_CACHE_KEY"


def test_source_frame_identity_is_deterministic() -> None:
    first = source_frame_identity(
        curve_id="curve_001", node_index=2, intent_id="i-7", d0=0.08)
    second = source_frame_identity(
        curve_id="curve_001", node_index=2, intent_id="i-7", d0=0.08)
    other = source_frame_identity(
        curve_id="curve_002", node_index=2, intent_id="i-7", d0=0.08)
    assert first == second
    assert first != other
    assert len(first) == 64


# ---------------------------------------------------------------------------
# 12. module-level API + typed refusal taxonomy
# ---------------------------------------------------------------------------
def test_module_level_build_context_matches_the_class(
        bridge: LatentBridge, tiny_view: dict) -> None:
    packet = _packet_from_view(tiny_view)
    direct = bridge.build_context(packet, tiny_view)
    loose = build_context(packet, tiny_view)
    for name in DECODER_CONTEXT_TENSOR_FIELDS:
        assert torch.equal(getattr(direct, name), getattr(loose, name)), name
    assert direct.bridge_spec_sha256 == loose.bridge_spec_sha256


def test_encoder_sha_is_recorded_and_validated(tiny_view: dict) -> None:
    packet = _packet_from_view(tiny_view)
    digest = "f" * 64
    context = LatentBridge(encoder_ckpt_sha256=digest).build_context(
        packet, tiny_view)
    assert context.encoder_ckpt_sha256 == digest
    with pytest.raises(V543BridgeError) as caught:
        LatentBridge(encoder_ckpt_sha256="not-a-digest")
    assert caught.value.code == "V543_BRIDGE_CKPT_SHA"


def test_refusals_carry_code_and_detail(bridge: LatentBridge) -> None:
    view = dict(_tiny_view())
    view.pop(NODE_INTENT_ROLE_KEY)
    with pytest.raises(V543BridgeError) as caught:
        bridge.reconstruct_changed_pairs(view)
    assert caught.value.code == "V543_BRIDGE_MISSING_KEY"
    assert caught.value.detail

    bad_dtype = dict(_tiny_view())
    bad_dtype[NODE_POS_KEY] = bad_dtype[NODE_POS_KEY].to(torch.float64)
    with pytest.raises(V543BridgeError) as caught:
        bridge.reconstruct_changed_pairs(bad_dtype)
    assert caught.value.code == "V543_BRIDGE_DTYPE"

    repeated = dict(_tiny_view())
    repeated[NODE_KEY_SET_KEY] = torch.tensor([0, 0, 4], dtype=torch.int64)
    with pytest.raises(V543BridgeError) as caught:
        bridge.reconstruct_changed_pairs(repeated)
    assert caught.value.code == "V543_BRIDGE_KEY_INDEX"
