"""V5.4.3 WP4 — Packet/Context cache CPU certificate (plan §5.7 :436–458,
§7.3 :735–746, §11.2 ``v543_feature_cache.py``).

Pinned here:

* the §5.7 six-component key is built by the FROZEN ``cache_key`` (never
  re-implemented) and a single drifted component changes the key;
* identity/mode/split refusals: ``joint_finetune`` forbids a cache, ``dev`` /
  ``confirm`` never enter one, a schema/identity mismatch refuses every hit;
* LRU semantics mirror the frozen ``_FeatureLRU``: recency on hit, oldest
  eviction on overflow, honest hit/miss counters;
* entries are stored DETACHED and CPU-resident (a cache read cannot backprop);
* ``split_context_by_graph`` / ``concat_contexts`` are the inverse of each
  other, and the decoder output on a concatenated context is bit-identical to
  the output on the whole-batch context — the property the S1 cache relies on."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3  # noqa: E402
from pes2ts_core.pesn.models.v543 import CoarseEncoder, LatentBridge  # noqa: E402
from pes2ts_core.pesn.models.v543.contracts import (  # noqa: E402
    PRECISION_VERSION_V543,
)
from pes2ts_core.pesn.models.v543.latent_bridge import (  # noqa: E402
    BRIDGE_SPEC_SHA256,
    CACHE_KEY_FIELDS,
)
from pes2ts_core.pesn.offline.data.v543_feature_cache import (  # noqa: E402
    CACHE_MODE_FROZEN_ENCODER,
    CACHE_MODE_JOINT_FINETUNE,
    HIT_RATE_TARGET_V543,
    RAM_LRU_PACKETS_V543,
    CacheIdentity,
    FrameRecord,
    PacketCache,
    V543_CACHE_SCHEMA,
    V543CacheError,
    assert_cache_split,
    batch_cache_key,
    canonicalize_key_edges,
    concat_contexts,
    contexts_equal,
    frame_cache_key,
    sha256_text,
    split_context_by_graph,
)
from tests.pesn._v543_fixtures_wp4 import (  # noqa: E402
    SynthMoveRefiner,
    batch_with_status,
    make_model,
)

ENCODER_SHA = "e" * 64


def _identity(**overrides) -> CacheIdentity:
    values = {"encoder_ckpt_sha256": ENCODER_SHA}
    values.update(overrides)
    return CacheIdentity(**values)


@pytest.fixture(scope="module")
def batch() -> dict:
    return batch_with_status(("ok",) * 4, jitter_seed=5)


@pytest.fixture(scope="module")
def encoded(batch: dict):
    view = forward_view_v3(dict(batch))
    encoder = CoarseEncoder()
    packet = encoder(view)
    context = LatentBridge(encoder_ckpt_sha256=ENCODER_SHA).build_context(
        packet, view)
    return view, packet, context


# ---------------------------------------------------------------------------
# §5.7 key / identity
# ---------------------------------------------------------------------------
def test_cache_constants_quote_the_campaign_yaml() -> None:
    assert RAM_LRU_PACKETS_V543 == 200_000
    assert HIT_RATE_TARGET_V543 == 0.80
    assert CACHE_KEY_FIELDS == (
        "encoder_ckpt_sha256", "bridge_spec_sha256", "input_feature_sha256",
        "source_frame_identity", "precision_version", "code_fingerprint")


def test_frame_key_has_six_components_in_order() -> None:
    identity = _identity()
    key = frame_cache_key(identity, curve_id="RXN_1", node_index=3,
                          intent_id=0, d0=1.5, split="inner_val")
    parts = key.split("|")
    assert len(parts) == len(CACHE_KEY_FIELDS)
    assert parts[0] == ENCODER_SHA
    assert parts[1] == BRIDGE_SPEC_SHA256
    assert parts[4] == PRECISION_VERSION_V543
    assert parts[5] == identity.code_fingerprint
    # deterministic and identity-sensitive
    assert key == frame_cache_key(identity, curve_id="RXN_1", node_index=3,
                                 intent_id=0, d0=1.5, split="inner_val")
    other = frame_cache_key(_identity(encoder_ckpt_sha256="f" * 64),
                            curve_id="RXN_1", node_index=3, intent_id=0,
                            d0=1.5, split="inner_val")
    assert other != key
    moved = frame_cache_key(identity, curve_id="RXN_1", node_index=4,
                            intent_id=0, d0=1.5, split="inner_val")
    assert moved != key


def test_joint_finetune_may_not_have_a_cache() -> None:
    with pytest.raises(V543CacheError) as error:
        _identity(mode=CACHE_MODE_JOINT_FINETUNE)
    assert error.value.code == "V543_CACHE_MODE"


def test_cache_identity_requires_every_component() -> None:
    for field in ("encoder_ckpt_sha256", "bridge_spec_sha256",
                  "precision_version", "code_fingerprint"):
        with pytest.raises(V543CacheError) as error:
            _identity(**{field: ""})
        assert error.value.code == "V543_CACHE_IDENTITY"
    with pytest.raises(V543CacheError) as error:
        _identity(schema="other_cache_v2")
    assert error.value.code == "V543_CACHE_SCHEMA"


def test_dev_and_confirm_never_enter_a_cache() -> None:
    for split in ("dev", "confirm"):
        with pytest.raises(V543CacheError) as error:
            assert_cache_split(split)
        assert error.value.code == "V543_CACHE_SPLIT"
    assert_cache_split("inner_train")
    assert_cache_split("inner_val")
    with pytest.raises(V543CacheError):
        assert_cache_split("probe_val")


def test_assert_cache_identity_accepts_the_same_identity() -> None:
    identity = _identity()
    cache = PacketCache(identity, capacity=4)
    header = identity.describe()
    assert cache.assert_cache_identity(header)["schema"] == V543_CACHE_SCHEMA


@pytest.mark.parametrize("field,value", [
    ("encoder_ckpt_sha256", "1" * 64),
    ("bridge_spec_sha256", "2" * 64),
    ("precision_version", "fp16_v2"),
    ("code_fingerprint", "other_code"),
    ("schema", "pesn_v543_packet_cache_v2"),
    ("mode", "joint_finetune"),
])
def test_assert_cache_identity_refuses_drift(field: str, value: str) -> None:
    cache = PacketCache(_identity(), capacity=4)
    header = dict(_identity().describe())
    header[field] = value
    with pytest.raises(V543CacheError) as error:
        cache.assert_cache_identity(header)
    assert error.value.code == "V543_CACHE_IDENTITY"
    assert field in error.value.detail


def test_batch_key_is_a_defensible_composition() -> None:
    keys = ["a|b|c|d|e|f", "a|b|g|h|e|f"]
    composed = batch_cache_key(keys)
    assert composed == sha256_text("v543_batch|" + "|".join(keys))
    assert batch_cache_key(list(reversed(keys))) != composed
    with pytest.raises(V543CacheError) as error:
        batch_cache_key([])
    assert error.value.code == "V543_CACHE_KEY_EMPTY"


# ---------------------------------------------------------------------------
# LRU semantics
# ---------------------------------------------------------------------------
def test_lru_evicts_the_oldest_and_counts(encoded) -> None:
    _view, packet, context = encoded
    cache = PacketCache(_identity(), capacity=2)
    parts = split_context_by_graph(context)
    assert len(parts) >= 3
    cache.put("k1", FrameRecord(context=parts[0]))
    cache.put("k2", FrameRecord(context=parts[1]))
    assert cache.size == 2
    assert cache.get("k1") is not None          # k1 becomes the MRU entry
    cache.put("k3", FrameRecord(context=parts[2]))
    assert cache.size == 2
    assert cache.evictions == 1
    assert cache.contains("k1") is True         # recency protected it
    assert cache.contains("k2") is False        # the oldest went out
    assert cache.contains("k3") is True
    assert cache.misses == 0 and cache.hits == 1


def test_lru_hit_miss_statistics(encoded) -> None:
    _view, packet, context = encoded
    cache = PacketCache(_identity(), capacity=8)
    cache.put("key", FrameRecord(context=context, packet=packet))
    assert cache.get("key") is not None
    assert cache.get("absent") is None
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["hit_rate_target"] == HIT_RATE_TARGET_V543
    assert stats["capacity"] == 8
    assert stats["mode"] == CACHE_MODE_FROZEN_ENCODER
    assert stats["identity"]["encoder_ckpt_sha256"] == ENCODER_SHA


def test_disabled_cache_never_pretends_to_hit(encoded) -> None:
    _view, packet, context = encoded
    cache = PacketCache(_identity(), capacity=4, enabled=False)
    assert cache.get("anything") is None
    assert cache.hits == 0 and cache.misses == 0
    with pytest.raises(V543CacheError) as error:
        cache.put("k", FrameRecord(context=context))
    assert error.value.code == "V543_CACHE_DISABLED"
    with pytest.raises(V543CacheError):
        cache.assert_cache_identity(_identity().describe())


def test_entries_are_stored_detached_and_cpu(encoded) -> None:
    _view, packet, context = encoded
    cache = PacketCache(_identity(), capacity=4)
    record = FrameRecord(context=context, packet=packet)
    stored = cache.put("k", record)
    for name in ("graph_s", "key_s", "key_v", "u0", "q0_raw"):
        assert stored.context.__getattribute__(name).requires_grad is False
        assert stored.context.__getattribute__(name).device.type == "cpu"
    assert stored.packet is not None
    assert stored.packet.z.requires_grad is False
    # the stored record still passes the WP1/WP2 validators
    stored.validate()


def test_get_or_build_and_lookup_many(encoded) -> None:
    _view, _packet, context = encoded
    cache = PacketCache(_identity(), capacity=8)
    calls = {"n": 0}

    def build() -> FrameRecord:
        calls["n"] += 1
        return FrameRecord(context=context)

    first = cache.get_or_build("k", build)
    assert calls["n"] == 1
    second = cache.get_or_build("k", build)
    assert calls["n"] == 1
    assert first.key == second.key == "k"
    assert cache.builds == 1
    hits = cache.lookup_many(["k", "missing"])
    assert hits[0] is not None and hits[1] is None
    with pytest.raises(V543CacheError) as error:
        cache.put("bad", "not a record")      # type: ignore[arg-type]
    assert error.value.code == "V543_CACHE_CONTENT"


def test_frame_record_requires_a_context(encoded) -> None:
    _view, packet, _context = encoded
    with pytest.raises(V543CacheError) as error:
        FrameRecord(context="nope")           # type: ignore[arg-type]
    assert error.value.code == "V543_CACHE_CONTENT"
    with pytest.raises(V543CacheError) as error:
        FrameRecord(context=None)             # type: ignore[arg-type]
    assert error.value.code == "V543_CACHE_CONTENT"


# ---------------------------------------------------------------------------
# split / concat (the cache's assembly primitives)
# ---------------------------------------------------------------------------
def test_split_is_the_inverse_of_concat(encoded) -> None:
    _view, _packet, context = encoded
    parts = split_context_by_graph(context)
    assert len(parts) == int(context.n_graphs)
    for part in parts:
        assert int(part.n_graphs) == 1
        assert int(part.key_ptr[0]) == 0
        assert int(part.pair_ptr[0]) == 0
        assert int(part.token_ptr[0]) == 0
    rebuilt = concat_contexts(parts)
    comparisons = contexts_equal(context, rebuilt)
    assert all(comparisons.values()), {
        key: value for key, value in comparisons.items() if not value}
    # the token layout is reproduced EXACTLY (no canonicalisation needed)
    assert torch.equal(context.token_index, rebuilt.token_index)
    assert torch.equal(context.token_type, rebuilt.token_type)
    assert torch.equal(context.token_graph, rebuilt.token_graph)
    assert torch.equal(context.token_ptr, rebuilt.token_ptr)


def test_canonicalize_key_edges_preserves_the_edge_set(encoded) -> None:
    _view, _packet, context = encoded
    canonical = canonicalize_key_edges(context)
    again = canonicalize_key_edges(canonical)
    assert torch.equal(canonical.key_edge_index, again.key_edge_index)
    assert torch.equal(canonical.key_edge_attr, again.key_edge_attr)
    if int(context.key_edge_index.shape[1]):
        left = {tuple(row) for row in context.key_edge_index.t().tolist()}
        right = {tuple(row) for row in canonical.key_edge_index.t().tolist()}
        assert left == right
        # the attribute rows follow their edges
        original = {
            tuple(edge): attr for edge, attr in zip(
                context.key_edge_index.t().tolist(),
                context.key_edge_attr.tolist())}
        moved = {
            tuple(edge): attr for edge, attr in zip(
                canonical.key_edge_index.t().tolist(),
                canonical.key_edge_attr.tolist())}
        assert original == moved


def test_concat_refuses_empty_and_multi_graph(encoded) -> None:
    _view, _packet, context = encoded
    with pytest.raises(V543CacheError):
        concat_contexts([])
    with pytest.raises(V543CacheError) as error:
        concat_contexts([context])
    assert error.value.code == "V543_CACHE_CONTEXT"


def test_decoder_output_is_identical_through_the_cache_path(encoded) -> None:
    """A context assembled from cached parts must not change the arithmetic."""
    _view, packet, context = encoded
    model = make_model(track="move", decoder=SynthMoveRefiner(),
                       decoder_id="M1")
    direct = model.forward_context(context)
    parts = split_context_by_graph(context)
    cached = model.forward_context(concat_contexts(parts))
    for key in ("u", "q", "z", "endpoint_logit", "order_score"):
        assert torch.equal(direct[key], cached[key]), key


def test_contexts_equal_names_the_drifted_field(encoded) -> None:
    _view, _packet, context = encoded
    import dataclasses

    fields = {
        name: getattr(context, name)
        for name in context.__dataclass_fields__}
    fields["key_s"] = fields["key_s"] + 1.0
    drifted = type(context)(**fields)
    result = contexts_equal(context, drifted)
    assert result["key_s"] is False
    assert result["graph_s"] is True
