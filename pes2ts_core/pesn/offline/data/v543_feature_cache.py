"""V5.4.3 Packet / Context cache — the S1 performance path (plan §5.7 :436–458,
§7.3 :735–746, §11.2 ``v543_feature_cache.py``; WP4).

Normative sources, quoted as implemented
----------------------------------------

* **§5.7 content rule (:439)** — "**缓存内容**：只有 ``DecoderContextV1`` 的
  tensor 字段 + schema/identity；**不含** ``y_teacher``、``q_true``、
  ``endpoint``、``frame_weight``、``sample_id``、``curve_id``."  A
  :class:`FrameRecord` stores the encoder packet's six tensors plus the
  context — both are supervision-free BY CONSTRUCTION (``CoarsePacket``
  carries ``z/edge_msg/node_h/u0/q0_raw/e0`` and ``DecoderContextV1`` is
  audited by ``assert_no_supervision``), and the entry key is a DIGEST, never a
  curve id (§5.7 :448: ``source_frame_identity`` is the canonical hash of
  "curve_id + node_index + intent_id + d0").
* **§5.7 cache key (:443–450)** — the six components in the plan's order are
  built by the FROZEN
  :func:`~pes2ts_core.pesn.models.v543.latent_bridge.cache_key`
  (``encoder_ckpt_sha256 | bridge_spec_sha256 | input_feature_sha256 |
  source_frame_identity | precision_version | code_fingerprint``).  This module
  never re-implements that composition — it calls it, so a WP2 change to the
  key cannot drift from the cache.
* **§5.7 rules (:452–458)** — (a) the cache is valid only in
  ``frozen_encoder`` mode; ``joint_finetune`` must build live and keep autograd
  (``V543_CACHE_MODE``); (b) only inner-train / inner-val curves enter (the
  cache types-refuses a ``dev``/``confirm`` label, ``V543_CACHE_SPLIT``);
  (c) per-curve sharding / RAM LRU; (d) identity, file SHA or schema mismatch ⇒
  refuse the hit and rebuild, never silently reuse
  (:meth:`PacketCache.assert_cache_identity`); (e) hit/miss/build counters go to
  the run records.
* **§7.3 (:737–746)** — "默认策略：live encoder + RAM LRU；缓存是可选加速",
  RAM LRU 200K packets (about 1 GB), hit-rate target ≥80 %, "缓存构建只允许
  inner-train + inner-val；dev/confirm 永不入缓存", "缓存键与身份见 §5.7；
  任何不匹配都拒绝命中", "``joint_finetune`` 模式禁用缓存".
  :data:`RAM_LRU_PACKETS_V543` / :data:`HIT_RATE_TARGET_V543` /
  :data:`DISK_MAX_GB_V543` / :data:`SHARD_CURVES_V543` quote the campaign YAML
  ``cache:`` block.

Design decisions and their basis
--------------------------------

1. **The cache stores the CONTEXT (the §5.7 delivery unit) and the Packet.**
   §5.7 says the cached content is the context; §7.3/§11.2 call the same object
   a "Packet cache".  Both are stored because the packet is what the Bridge
   consumed and what the encoder audit reads back — and neither can carry a
   label, so storing both cannot violate §5.7 :439 (the refusals below re-check
   the supervision-free property on every construction).
2. **LRU semantics mirror the frozen ``_FeatureLRU``**
   (``key_atom_v53mp_train_v3.py`` :737–797): ``OrderedDict`` storage, a hit
   pops and re-inserts (true recency), overflow evicts the OLDEST entry
   (``popitem(last=False)``), plain integer counters.  Reusing the frozen
   semantics keeps V543's hit-rate numbers comparable to V5.3-MP's.
3. **Entries are detached and CPU-resident.**  A frozen-encoder cache is read
   without autograd (§7.3 :744 "``joint_finetune`` 模式禁用缓存，实时前向 +
   完整 autograd"), so :meth:`PacketCache.put` detaches — a caller cannot
   accidentally backprop through a stale graph.
4. **``split_context_by_graph`` / ``concat_contexts`` are the cache's assembly
   primitives.**  §7.3 caches per frame (one graph per frame) while the decoder
   consumes one batch context; the two helpers are exact inverses (proven in
   the WP4 tests).  Keeping them here leaves ``model.py``'s forward flow free
   of cache-shaped code.

Discipline: no V5.4.2/V5.3-MP file modified, no ``torch.compile``, ``Final``
constants, Python 3.12."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Final, Iterable, Mapping, Sequence

import torch

from pes2ts_core.pesn.models.v543.contracts import (
    DECODER_CONTEXT_FIELDS,
    DECODER_CONTEXT_IDENTITY_FIELDS,
    DECODER_CONTEXT_SCHEMA_V543,
    PRECISION_VERSION_V543,
    CoarsePacket,
    DecoderContextV1,
    V543Error,
    assert_no_supervision,
    context_field_names,
    validate_coarse_packet,
    validate_decoder_context,
)
from pes2ts_core.pesn.models.v543.latent_bridge import (
    BRIDGE_SPEC_SHA256,
    CACHE_KEY_FIELDS,
    cache_key,
    source_frame_identity,
)

#: Schema id of a cache entry / cache description.
V543_CACHE_SCHEMA: Final[str] = "pesn_v543_packet_cache_v1"

#: The only mode in which a cache is legal (plan §5.7 :454, §7.3 :744).
CACHE_MODE_FROZEN_ENCODER: Final[str] = "frozen_encoder_only"
#: The mode that FORBIDS the cache (live forward + full autograd).
CACHE_MODE_JOINT_FINETUNE: Final[str] = "joint_finetune"
CACHE_MODES_V543: Final[tuple[str, ...]] = (
    CACHE_MODE_FROZEN_ENCODER, CACHE_MODE_JOINT_FINETUNE)

#: Campaign YAML ``cache:`` block, quoted verbatim (§7.3 :739–740).
RAM_LRU_PACKETS_V543: Final[int] = 200_000
DISK_MAX_GB_V543: Final[int] = 30
SHARD_CURVES_V543: Final[int] = 256
HIT_RATE_TARGET_V543: Final[float] = 0.80
#: Splits a cache may be built from (plan §5.7 :455, §7.3 :742).
CACHE_ALLOWED_SPLITS_V543: Final[tuple[str, ...]] = ("inner_train", "inner_val")
#: Splits that may NEVER enter a cache (plan §5.7 :455 "dev/confirm 永不入缓存").
CACHE_FORBIDDEN_SPLITS_V543: Final[tuple[str, ...]] = ("dev", "confirm")

#: Default code-fingerprint component of the §5.7 key.  A runner may pass a
#: stronger digest; this default names the cache implementation itself, so a
#: change HERE invalidates every entry this implementation produced.
CODE_FINGERPRINT_V543: Final[str] = "pesn_v543_feature_cache_v1"

#: The six encoder-packet fields a cached packet may carry (plan §5.7 :439).
PACKET_FIELDS_V543: Final[tuple[str, ...]] = (
    "z", "edge_msg", "node_h", "u0", "q0_raw", "e0")


class V543CacheError(V543Error):
    """Cache identity / mode / content refusal (typed, fail-closed)."""


# ---------------------------------------------------------------------------
# identity block (§5.7 :443–450)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CacheIdentity:
    """The frozen identity of one cache (the six §5.7 components' sources).

    ``encoder_ckpt_sha256`` / ``bridge_spec_sha256`` / ``precision_version`` /
    ``code_fingerprint`` are constant for a run; ``input_feature_sha256`` and
    ``source_frame_identity`` vary per frame and are supplied per lookup.
    """

    encoder_ckpt_sha256: str
    bridge_spec_sha256: str = BRIDGE_SPEC_SHA256
    precision_version: str = PRECISION_VERSION_V543
    code_fingerprint: str = CODE_FINGERPRINT_V543
    mode: str = CACHE_MODE_FROZEN_ENCODER
    schema: str = V543_CACHE_SCHEMA

    def __post_init__(self) -> None:
        if self.mode not in CACHE_MODES_V543:
            raise V543CacheError(
                "V543_CACHE_MODE",
                f"cache mode must be one of {list(CACHE_MODES_V543)}, got "
                f"{self.mode!r}")
        if self.mode == CACHE_MODE_JOINT_FINETUNE:
            raise V543CacheError(
                "V543_CACHE_MODE",
                "a cache may not be constructed in joint_finetune mode; that "
                "mode needs a live forward with full autograd (plan §5.7 :454, "
                "§7.3 :744)")
        for name in ("encoder_ckpt_sha256", "bridge_spec_sha256",
                     "precision_version", "code_fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise V543CacheError(
                    "V543_CACHE_IDENTITY",
                    f"cache identity component {name!r} must be a non-empty "
                    f"string (plan §5.7 :443–450), got {value!r}")
        if self.schema != V543_CACHE_SCHEMA:
            raise V543CacheError(
                "V543_CACHE_SCHEMA",
                f"cache schema must be {V543_CACHE_SCHEMA!r}, got "
                f"{self.schema!r}")

    def frame_key(
            self,
            *,
            input_feature_sha256: str,
            source_frame_identity: str,
    ) -> str:
        """The §5.7 cache key of ONE frame (delegated to the frozen builder)."""
        return cache_key(
            encoder_ckpt_sha256=self.encoder_ckpt_sha256,
            bridge_spec_sha256=self.bridge_spec_sha256,
            input_feature_sha256=input_feature_sha256,
            source_frame_identity=source_frame_identity,
            precision_version=self.precision_version,
            code_fingerprint=self.code_fingerprint,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "encoder_ckpt_sha256": self.encoder_ckpt_sha256,
            "bridge_spec_sha256": self.bridge_spec_sha256,
            "precision_version": self.precision_version,
            "code_fingerprint": self.code_fingerprint,
            "mode": self.mode,
            "cache_key_fields": list(CACHE_KEY_FIELDS),
        }


def frame_cache_key(
        identity: CacheIdentity,
        *,
        curve_id: str,
        node_index: int,
        intent_id: object,
        d0: float,
        input_feature_sha256: str = "",
        split: str | None = None,
) -> str:
    """One frame's §5.7 key from its production identity (convenience wrapper).

    ``input_feature_sha256`` defaults to a digest of the frame identity inputs
    themselves — the feature bytes are a PURE function of the record (frozen
    ``_FeatureLRU``, ``key_atom_v53mp_train_v3.py`` :771–785), so recording the
    inputs is recording the features' identity.  A caller holding the actual
    feature bytes may pass its own digest.
    """
    if split is not None:
        assert_cache_split(split)
    frame_identity = source_frame_identity(
        curve_id=curve_id, node_index=int(node_index),
        intent_id=intent_id, d0=float(d0))
    digest = (str(input_feature_sha256)
              or sha256_text(f"v543_frame_features|{frame_identity}"))
    return identity.frame_key(
        input_feature_sha256=digest, source_frame_identity=frame_identity)


def batch_cache_key(frame_keys: Sequence[str]) -> str:
    """Compose per-frame §5.7 keys into ONE batch delivery key.

    The plan's cache key is per frame (§5.7 :443–450); a decoder batch consumes
    a context of G graphs and the S1 loop hands the encoder a batch.  The
    composition is DEFINED here (not implied) so it stays auditable: the
    ordered tuple of frame keys, hashed.  Reordering a batch changes the key —
    correct, because a batch context's row order is part of its identity.
    """
    keys = [str(k) for k in frame_keys]
    if not keys:
        raise V543CacheError(
            "V543_CACHE_KEY_EMPTY",
            "a batch cache key needs at least one frame key")
    return sha256_text("v543_batch|" + "|".join(keys))


def sha256_text(text: str) -> str:
    """Hex sha256 of a UTF-8 string (canonical, dependency-free)."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def assert_cache_split(split: str) -> None:
    """Refuse a cache build outside inner-train / inner-val (§5.7 :455)."""
    text = str(split)
    if text in CACHE_FORBIDDEN_SPLITS_V543:
        raise V543CacheError(
            "V543_CACHE_SPLIT",
            f"the {text!r} split may never enter a cache (plan §5.7 :455 / "
            f"§7.3 :742); only {list(CACHE_ALLOWED_SPLITS_V543)} are "
            "cacheable")
    if text not in CACHE_ALLOWED_SPLITS_V543:
        raise V543CacheError(
            "V543_CACHE_SPLIT",
            f"unknown split {text!r}; a cache is built from "
            f"{list(CACHE_ALLOWED_SPLITS_V543)} only")


# ---------------------------------------------------------------------------
# one cached frame
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FrameRecord:
    """One cached delivery unit: a ``DecoderContextV1`` (+ its Packet if held).

    The CONTEXT is the mandatory member — it is what the decoder consumes and
    what plan §5.7 :439 names as the cached content.  ``packet`` is optional
    because a per-frame cache entry built by the S1 loop owns only the context
    (the encoder packet's node/edge rows are batch-level, not frame-level);
    tooling that holds a whole packet (parity certificates) may store it too.
    Both members are supervision-free by construction, and the record is stored
    DETACHED/CPU by :meth:`PacketCache.put` (:meth:`detached_cpu`).
    """

    context: DecoderContextV1
    packet: CoarsePacket | None = None
    key: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.context, DecoderContextV1):
            raise V543CacheError(
                "V543_CACHE_CONTENT",
                f"FrameRecord.context must be a DecoderContextV1, got "
                f"{type(self.context).__name__}")
        if self.packet is not None:
            if not isinstance(self.packet, CoarsePacket):
                raise V543CacheError(
                    "V543_CACHE_CONTENT",
                    f"FrameRecord.packet must be a CoarsePacket or None, got "
                    f"{type(self.packet).__name__}")
            _assert_supervision_free_packet(self.packet)
        assert_no_supervision(self.context)

    @property
    def n_graphs(self) -> int:
        return int(self.context.n_graphs)

    def detached_cpu(self) -> "FrameRecord":
        """A detached, CPU copy (what the cache stores, behaviour 3)."""
        packet = None
        if self.packet is not None:
            packet = CoarsePacket(**{
                name: _detach_cpu(getattr(self.packet, name))
                for name in PACKET_FIELDS_V543})
        fields: dict[str, Any] = {
            name: getattr(self.context, name)
            for name in DECODER_CONTEXT_FIELDS}
        for name in context_field_names(self.context):
            value = fields[name]
            fields[name] = (value if isinstance(value, str)
                            else _detach_cpu(value))
        return replace(self, packet=packet,
                       context=DecoderContextV1(**fields))

    def validate(self) -> "FrameRecord":
        """Re-run the WP1/WP2 validators on a record read from the cache."""
        ctx = self.context
        if self.packet is not None:
            validate_coarse_packet(
                self.packet, n_graphs=int(ctx.n_graphs),
                n_nodes=int(self.packet.node_h.shape[0]),
                n_keys=int(ctx.n_keys),
                n_edges=int(self.packet.edge_msg.shape[0]),
                z_dim=int(ctx.graph_s.shape[1]),
                hidden=int(ctx.key_s.shape[1]))
        validate_decoder_context(ctx)
        return self

    def describe(self) -> dict[str, Any]:
        return {
            "schema": V543_CACHE_SCHEMA,
            "key": self.key,
            "n_graphs": int(self.context.n_graphs),
            "n_keys": int(self.context.n_keys),
            "n_pairs": int(self.context.n_pairs),
            "encoder_ckpt_sha256": self.context.encoder_ckpt_sha256,
            "bridge_spec_sha256": self.context.bridge_spec_sha256,
            "precision_version": self.context.precision_version,
        }


def _detach_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu").clone()


def _assert_supervision_free_packet(packet: CoarsePacket) -> None:
    """A cached packet may carry the six encoder tensors and nothing else."""
    extra = set(context_field_names(packet)) - set(PACKET_FIELDS_V543)
    if extra:                              # pragma: no cover - frozen dataclass
        raise V543CacheError(
            "V543_CACHE_CONTENT",
            f"CoarsePacket gained field(s) {sorted(extra)}; a cached packet "
            "may carry the six encoder outputs only (plan §5.7 :439)")


# ---------------------------------------------------------------------------
# the LRU
# ---------------------------------------------------------------------------
class PacketCache:
    """Bounded LRU of :class:`FrameRecord` keyed by the §5.7 frame key.

    Mirrors the frozen ``_FeatureLRU`` (``key_atom_v53mp_train_v3.py`` :737):
    ``OrderedDict`` storage, a hit pops + re-inserts (recency), overflow evicts
    the OLDEST entry, plain integer counters.
    """

    def __init__(
            self,
            identity: CacheIdentity,
            *,
            capacity: int = RAM_LRU_PACKETS_V543,
            enabled: bool = True,
            shard_curves: int = SHARD_CURVES_V543,
    ) -> None:
        if int(capacity) < 1:
            raise V543CacheError(
                "V543_CACHE_CAPACITY",
                f"capacity must be >= 1, got {capacity}")
        if not isinstance(identity, CacheIdentity):
            raise V543CacheError(
                "V543_CACHE_IDENTITY",
                f"PacketCache needs a CacheIdentity, got "
                f"{type(identity).__name__} (plan §5.7 :443–450)")
        self.identity = identity
        self.capacity = int(capacity)
        self.enabled = bool(enabled)
        self.shard_curves = int(shard_curves)
        self._data: "OrderedDict[str, FrameRecord]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._builds = 0
        self._build_seconds = 0.0

    # -- identity -----------------------------------------------------------
    def assert_cache_identity(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Refuse a cache built under a different identity (plan §5.7 :457).

        ``record`` is a serialised cache header (identity block + schema).  Any
        drift of ``encoder_ckpt_sha256`` / ``bridge_spec_sha256`` /
        ``precision_version`` / ``code_fingerprint`` / ``mode`` / ``schema`` is
        a typed ``V543_CACHE_IDENTITY`` refusal — "不得静默复用".
        """
        self._assert_enabled("an identity check")
        if not isinstance(record, Mapping):
            raise V543CacheError(
                "V543_CACHE_IDENTITY",
                f"cache identity record must be a mapping, got "
                f"{type(record).__name__}")
        expected = self.identity.describe()
        mismatches = sorted(
            key for key in ("schema", "encoder_ckpt_sha256",
                            "bridge_spec_sha256", "precision_version",
                            "code_fingerprint", "mode")
            if str(record.get(key)) != str(expected.get(key)))
        if mismatches:
            detail = "; ".join(
                f"{key}: cache={record.get(key)!r} run={expected.get(key)!r}"
                for key in mismatches)
            raise V543CacheError(
                "V543_CACHE_IDENTITY",
                "cache identity mismatch — refusing every hit and rebuilding "
                "(plan §5.7 :457): " + detail)
        declared_fields = record.get("cache_key_fields")
        if declared_fields is not None and list(
                declared_fields) != list(CACHE_KEY_FIELDS):
            raise V543CacheError(
                "V543_CACHE_IDENTITY",
                "cache key field order drifted from the plan §5.7 sequence: "
                f"{declared_fields!r}")
        return dict(record)

    # -- counters -----------------------------------------------------------
    def _assert_enabled(self, what: str) -> None:
        if not self.enabled:
            raise V543CacheError(
                "V543_CACHE_DISABLED",
                f"cache is disabled; {what} is not available (a disabled "
                "cache never pretends to hit)")

    @property
    def hits(self) -> int:
        return int(self._hits)

    @property
    def misses(self) -> int:
        return int(self._misses)

    @property
    def evictions(self) -> int:
        return int(self._evictions)

    @property
    def builds(self) -> int:
        return int(self._builds)

    @property
    def hit_rate(self) -> float:
        total = int(self._hits) + int(self._misses)
        return 0.0 if total == 0 else float(self._hits) / float(total)

    @property
    def size(self) -> int:
        return len(self._data)

    def contains(self, key: str) -> bool:
        return str(key) in self._data

    # -- get / put ----------------------------------------------------------
    def get(self, key: str) -> FrameRecord | None:
        """LRU lookup; ``None`` (and a miss) when absent or disabled."""
        if not self.enabled:
            return None
        record = self._data.pop(str(key), None)
        if record is None:
            self._misses += 1
            return None
        self._data[str(key)] = record          # re-insert at the MRU end
        self._hits += 1
        return record

    def put(self, key: str, record: FrameRecord) -> FrameRecord:
        """Store a DETACHED CPU copy at the MRU end, evicting the LRU entry."""
        self._assert_enabled("a put")
        if not isinstance(record, FrameRecord):
            raise V543CacheError(
                "V543_CACHE_CONTENT",
                f"cache entries must be FrameRecord, got "
                f"{type(record).__name__}")
        stored = replace(record.detached_cpu(), key=str(key))
        self._data.pop(str(key), None)
        self._data[str(key)] = stored
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)
            self._evictions += 1
        return stored

    def put_many(self, items: Iterable[tuple[str, FrameRecord]]) -> int:
        count = 0
        for key, record in items:
            self.put(key, record)
            count += 1
        return count

    def lookup_many(self, keys: Sequence[str]) -> list[FrameRecord | None]:
        """One pass over a batch's frame keys (hit/miss split)."""
        return [self.get(str(key)) for key in keys]

    def note_builds(self, n_builds: int, seconds: float = 0.0) -> None:
        """Record encoder/bridge build work (run-records counters, §7.2 :730)."""
        self._builds += int(n_builds)
        self._build_seconds += float(seconds)

    def get_or_build(self, key: str, build: Any) -> FrameRecord:
        """Hit, else ``build()`` (a 0-arg callable) and store its record."""
        hit = self.get(str(key))
        if hit is not None:
            return hit
        record = build()
        self.note_builds(1)
        return self.put(str(key), record)

    def clear(self) -> None:
        self._data.clear()

    # -- statistics ---------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Counters for ``run_records.jsonl`` (plan §7.2 :730 / §5.7 :458)."""
        return {
            "schema": V543_CACHE_SCHEMA,
            "mode": self.identity.mode,
            "enabled": bool(self.enabled),
            "capacity": int(self.capacity),
            "shard_curves": int(self.shard_curves),
            "size": len(self._data),
            "hits": int(self._hits),
            "misses": int(self._misses),
            "evictions": int(self._evictions),
            "builds": int(self._builds),
            "build_seconds": round(float(self._build_seconds), 6),
            "hit_rate": self.hit_rate,
            "hit_rate_target": HIT_RATE_TARGET_V543,
            "identity": self.identity.describe(),
        }

    def describe(self) -> dict[str, Any]:
        payload = self.stats()
        payload.update({
            "ram_lru_packets": RAM_LRU_PACKETS_V543,
            "disk_max_gb": DISK_MAX_GB_V543,
            "allowed_splits": list(CACHE_ALLOWED_SPLITS_V543),
            "forbidden_splits": list(CACHE_FORBIDDEN_SPLITS_V543),
            "content": "CoarsePacket (six tensors) + DecoderContextV1",
        })
        return payload


# ---------------------------------------------------------------------------
# per-graph split / concat (the cache's assembly primitives)
# ---------------------------------------------------------------------------
def _key_edge_mask(key_edge_index: torch.Tensor, key_graph: torch.Tensor,
                   graph: int) -> torch.Tensor:
    if int(key_edge_index.shape[1]) == 0:
        return torch.zeros(0, dtype=torch.bool)
    rows = key_edge_index[0].long()
    return key_graph[rows] == int(graph)


def _rebase_token_index(token_index: torch.Tensor, token_type: torch.Tensor,
                        k0: int, p0: int) -> torch.Tensor:
    out = token_index.clone()
    types = token_type.to(torch.int64)
    out[types == 1] = out[types == 1] - int(k0)
    out[types == 2] = out[types == 2] - int(p0)
    return out


def canonicalize_key_edges(ctx: DecoderContextV1) -> DecoderContextV1:
    """Return ``ctx`` with ``key_edge_*`` columns sorted by ``(src, dst)``.

    ``key_edge_index``/``key_edge_attr`` describe a SET of key-key edges (§5.5),
    so two contexts of the same edge set may differ only in row order.  Sorting
    both tensors with the SAME permutation makes a bit-exact comparison
    (``torch.equal``) meaningful — used by the cache roundtrip audit and by any
    caller that compares a cached context against a live one.
    """
    if not isinstance(ctx, DecoderContextV1):
        raise V543CacheError(
            "V543_CACHE_CONTEXT",
            f"canonicalize_key_edges needs a DecoderContextV1, got "
            f"{type(ctx).__name__}")
    n_edges = int(ctx.key_edge_index.shape[1])
    if n_edges <= 1:
        return ctx
    lo = ctx.key_edge_index[0].to(torch.int64)
    hi = ctx.key_edge_index[1].to(torch.int64)
    order = torch.argsort(lo * (int(ctx.n_keys) + 1) + hi, stable=True)
    fields = {name: getattr(ctx, name) for name in DECODER_CONTEXT_FIELDS}
    fields["key_edge_index"] = ctx.key_edge_index[:, order]
    fields["key_edge_attr"] = ctx.key_edge_attr[order]
    return DecoderContextV1(**fields)


def contexts_equal(left: DecoderContextV1, right: DecoderContextV1, *,
                   canonicalize_edges: bool = True) -> dict[str, bool]:
    """Field-by-field ``torch.equal`` audit of two contexts.

    ``canonicalize_edges`` (default) sorts the key-edge rows first, because the
    edge set's order is not part of its meaning (§5.5).  Returns one boolean per
    field so a failing comparison NAMES the drifted field instead of raising a
    bare ``AssertionError``.
    """
    lhs = canonicalize_key_edges(left) if canonicalize_edges else left
    rhs = canonicalize_key_edges(right) if canonicalize_edges else right
    result: dict[str, bool] = {}
    for name in DECODER_CONTEXT_FIELDS:
        a, b = getattr(lhs, name), getattr(rhs, name)
        if isinstance(a, str) or isinstance(b, str):
            result[name] = a == b
            continue
        result[name] = (tuple(a.shape) == tuple(b.shape)
                        and a.dtype == b.dtype
                        and bool(torch.equal(a, b)))
    return result


def split_context_by_graph(ctx: DecoderContextV1) -> list[DecoderContextV1]:
    """Split a batch context into ONE single-graph context per graph.

    The inverse of :func:`concat_contexts`: every tensor field is sliced by the
    frozen prefix sums (``key_ptr`` / ``pair_ptr`` / ``token_ptr``, plan §5.2
    形状约束) and each part's pointers restart at 0.

    ``key_edge_*`` is a SET, not a sequence (plan §5.5 :395–405): the batch
    context stores the two directions in two passes, while a single-graph part
    stores each graph's edges together.  Membership and the edge↔attribute
    alignment are preserved exactly; only the ROW ORDER of ``key_edge_*``
    changes, which :func:`canonicalize_key_edges` makes comparable
    (``torch.equal``-able) again.
    """
    if not isinstance(ctx, DecoderContextV1):
        raise V543CacheError(
            "V543_CACHE_CONTEXT",
            f"split_context_by_graph needs a DecoderContextV1, got "
            f"{type(ctx).__name__}")
    parts: list[DecoderContextV1] = []
    for graph in range(int(ctx.n_graphs)):
        k0, k1 = int(ctx.key_ptr[graph]), int(ctx.key_ptr[graph + 1])
        p0, p1 = int(ctx.pair_ptr[graph]), int(ctx.pair_ptr[graph + 1])
        t0, t1 = int(ctx.token_ptr[graph]), int(ctx.token_ptr[graph + 1])
        edge_mask = _key_edge_mask(ctx.key_edge_index, ctx.key_graph, graph)
        parts.append(DecoderContextV1(
            graph_s=ctx.graph_s[graph:graph + 1],
            key_s=ctx.key_s[k0:k1],
            key_v=ctx.key_v[k0:k1],
            u0=ctx.u0[k0:k1],
            key_graph=ctx.key_graph[k0:k1] - graph,
            key_ptr=torch.tensor([0, k1 - k0], dtype=torch.int64),
            key_edge_index=(ctx.key_edge_index[:, edge_mask] - k0
                            if bool(edge_mask.any())
                            else torch.zeros(2, 0, dtype=torch.int64)),
            key_edge_attr=(ctx.key_edge_attr[edge_mask]
                           if bool(edge_mask.any())
                           else torch.zeros(0, ctx.key_edge_attr.shape[1])),
            pair_s=ctx.pair_s[p0:p1],
            intent_pair_index=ctx.intent_pair_index[:, p0:p1] - k0,
            pair_graph=ctx.pair_graph[p0:p1] - graph,
            pair_ptr=torch.tensor([0, p1 - p0], dtype=torch.int64),
            token_index=_rebase_token_index(
                ctx.token_index[t0:t1], ctx.token_type[t0:t1], k0, p0),
            token_type=ctx.token_type[t0:t1],
            token_graph=ctx.token_graph[t0:t1] - graph,
            token_ptr=torch.tensor([0, t1 - t0], dtype=torch.int64),
            q0_raw=ctx.q0_raw[graph:graph + 1],
            endpoint0_logit=ctx.endpoint0_logit[graph:graph + 1],
            schema=ctx.schema,
            bridge_spec_sha256=ctx.bridge_spec_sha256,
            encoder_ckpt_sha256=ctx.encoder_ckpt_sha256,
            precision_version=ctx.precision_version,
        ))
    return parts


def concat_contexts(parts: Sequence[DecoderContextV1]) -> DecoderContextV1:
    """Concatenate single-graph contexts into one batch context (§5.2 ptrs)."""
    if not parts:
        raise V543CacheError(
            "V543_CACHE_CONTEXT", "concat_contexts needs at least one context")
    for part in parts:
        if not isinstance(part, DecoderContextV1):
            raise V543CacheError(
                "V543_CACHE_CONTEXT",
                "concat_contexts needs DecoderContextV1 parts, got "
                f"{type(part).__name__}")
        if int(part.n_graphs) != 1:
            raise V543CacheError(
                "V543_CACHE_CONTEXT",
                "concat_contexts takes single-graph contexts; got one with "
                f"{int(part.n_graphs)} graphs")
    identity = parts[0]
    for part in parts[1:]:
        for name in DECODER_CONTEXT_IDENTITY_FIELDS:
            if getattr(part, name) != getattr(identity, name):
                raise V543CacheError(
                    "V543_CACHE_IDENTITY",
                    f"cannot concatenate contexts with different {name!r} "
                    f"({getattr(part, name)!r} vs "
                    f"{getattr(identity, name)!r}) — plan §5.7 :457")

    key_ptr = [0]
    pair_ptr = [0]
    token_ptr = [0]
    for part in parts:
        key_ptr.append(key_ptr[-1] + int(part.n_keys))
        pair_ptr.append(pair_ptr[-1] + int(part.n_pairs))
        token_ptr.append(token_ptr[-1] + int(part.n_tokens))
    key_offsets = [int(x) for x in key_ptr[:-1]]
    pair_offsets = [int(x) for x in pair_ptr[:-1]]

    token_index_parts: list[torch.Tensor] = []
    token_type_parts: list[torch.Tensor] = []
    token_graph_parts: list[torch.Tensor] = []
    key_edges: list[torch.Tensor] = []
    key_attrs: list[torch.Tensor] = []
    for graph, part in enumerate(parts):
        types = part.token_type.to(torch.int64)
        index = part.token_index.clone()
        index[types == 1] = index[types == 1] + key_offsets[graph]
        index[types == 2] = index[types == 2] + pair_offsets[graph]
        token_index_parts.append(index)
        token_type_parts.append(part.token_type)
        token_graph_parts.append(part.token_graph + graph)
        if int(part.key_edge_index.shape[1]):
            key_edges.append(part.key_edge_index + key_offsets[graph])
            key_attrs.append(part.key_edge_attr)

    def _cat(name: str) -> torch.Tensor:
        return torch.cat([getattr(part, name) for part in parts], dim=0)

    pair_rows = sum(int(part.n_pairs) for part in parts)
    return DecoderContextV1(
        graph_s=_cat("graph_s"),
        key_s=_cat("key_s"),
        key_v=_cat("key_v"),
        u0=_cat("u0"),
        key_graph=torch.cat([part.key_graph + graph
                             for graph, part in enumerate(parts)]),
        key_ptr=torch.tensor(key_ptr, dtype=torch.int64),
        key_edge_index=(torch.cat(key_edges, dim=1) if key_edges
                        else torch.zeros(2, 0, dtype=torch.int64)),
        key_edge_attr=(torch.cat(key_attrs, dim=0) if key_attrs
                       else torch.zeros(0, int(parts[0].key_edge_attr.shape[1]))),
        pair_s=_cat("pair_s"),
        intent_pair_index=(torch.cat(
            [part.intent_pair_index + key_offsets[graph]
             for graph, part in enumerate(parts)], dim=1)
            if pair_rows or len(parts) > 1
            else torch.zeros(2, 0, dtype=torch.int64)),
        pair_graph=torch.cat([part.pair_graph + graph
                              for graph, part in enumerate(parts)]),
        pair_ptr=torch.tensor(pair_ptr, dtype=torch.int64),
        token_index=torch.cat(token_index_parts),
        token_type=torch.cat(token_type_parts),
        token_graph=torch.cat(token_graph_parts),
        token_ptr=torch.tensor(token_ptr, dtype=torch.int64),
        q0_raw=_cat("q0_raw"),
        endpoint0_logit=_cat("endpoint0_logit"),
        schema=identity.schema,
        bridge_spec_sha256=identity.bridge_spec_sha256,
        encoder_ckpt_sha256=identity.encoder_ckpt_sha256,
        precision_version=identity.precision_version,
    )


__all__ = [
    "CACHE_ALLOWED_SPLITS_V543",
    "CACHE_FORBIDDEN_SPLITS_V543",
    "CACHE_MODES_V543",
    "CACHE_MODE_FROZEN_ENCODER",
    "CACHE_MODE_JOINT_FINETUNE",
    "CODE_FINGERPRINT_V543",
    "DISK_MAX_GB_V543",
    "HIT_RATE_TARGET_V543",
    "PACKET_FIELDS_V543",
    "RAM_LRU_PACKETS_V543",
    "SHARD_CURVES_V543",
    "CacheIdentity",
    "FrameRecord",
    "PacketCache",
    "V543_CACHE_SCHEMA",
    "V543CacheError",
    "assert_cache_split",
    "batch_cache_key",
    "concat_contexts",
    "frame_cache_key",
    "sha256_text",
    "split_context_by_graph",
]
