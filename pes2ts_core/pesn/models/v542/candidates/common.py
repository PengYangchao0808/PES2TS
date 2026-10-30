"""V5.4.2 shared candidate scaffolding for lanes A1–A5 (WP2 lane P2-0;
ADR-0033 §2.3; config ``training_contract.init_policy``).

This module is the SINGLE import surface every candidate lane builds on.
What it freezes (stable for A1, A1-C, A2, A3, A4, A5, A5-C):

* **R3 warm-start identity** — :data:`R3_CHECKPOINT_PATH` /
  :data:`R3_CHECKPOINT_SHA256` with :func:`load_r3_backbone_state`:
  SHA-256 is computed over the FILE BYTES; a missing file or a digest
  drift is a typed refusal (codes ``V542_R3_INIT_MISSING`` /
  ``V542_R3_INIT_SHA256``; payload shape problems are
  ``V542_R3_INIT_PAYLOAD``), and the pinned 37-key ``payload["model"]``
  backbone state dict is returned.  No fallback path, no re-hash-once
  caching — every call re-verifies (config ``verify_on_validate: true``);
* **the frozen seed recipe** — :data:`INIT_SEED_TEMPLATE`,
  :data:`INIT_SOURCE_ID_OVERRIDES`, :func:`seed_init`
  (``int(sha256(...).digest()[:8], "big") % 2147483647``, byte-identical
  to the frozen ``hexdigest[:16]`` algorithm) and
  :func:`resolve_init_source_id` (A1-C→A1, A5-C→A5 attribution overrides);
* **the frozen construction protocol** (config
  ``init_policy.construction_order`` 1→6): :func:`construction_scope`
  wraps the WHOLE build in ``torch.random.fork_rng(devices=[])`` +
  ``manual_seed(seed_init)``; inside it the caller (1) builds fresh
  modules, (2) applies :func:`apply_r3_mapping`, (3) zero-initializes
  special parameters via :func:`zero_init_`, (4) loads the SHA-verified
  R3 ``w_v`` into the shared decoder via :func:`apply_r3_w_v` (or
  :func:`apply_shared_w_v` when the backbone adopts the decoder's
  ``w_v`` object), (5) builds the process head with
  :func:`build_process_head`; after leaving the scope the caller MUST
  call :func:`finish_training_seed` (step 6) before training;
* **the strict R3 semantic mapping audit** — :func:`apply_r3_mapping`
  is explicit and pre-registered: ``mapping`` is ``{target_key:
  source_key}``, there are NO runtime shape heuristics and NO ``copy_``
  before every category has been audited clean;
* **budget / contract gates** — :func:`assert_candidate_budget` (the
  70k active-parameter hard cap) and :func:`check_backbone_contract`
  (``BackboneOutput`` shapes, fp32, finiteness, a LIVE autograd graph,
  and no torch.compile anywhere in the tree).

Audit semantics — STRICTER reading of config ``init_policy.mapping_audit``
("every source key consumed exactly once ... no empty slots"): EVERY key
of ``source_state`` must be accounted for EXACTLY once — consumed by one
``mapping`` entry, by one declared ``dual_write`` pair (the declared
exception: that source is then written twice), or listed in
``consumed_source_keys`` for consumption that happens OUTSIDE this call
(e.g. ``w_v.*`` by :func:`apply_r3_w_v` / :func:`apply_shared_w_v` —
the mapping call runs BEFORE construction step 4 loads the decoder
``w_v`` from the SAME resolved source, so those keys are declared up
front and genuinely consumed right after the audit; or payload keys a
candidate deliberately does not map, such as R3's third-layer keys for
a 2-layer candidate).  A payload key nobody accounts for is
``unexpected``; a source consumed more times than its declarations allow
is an undeclared double consumption.  The four loader-mirroring audit
categories ``missing`` / ``unexpected`` / ``shape_mismatch`` /
``dtype_mismatch`` must ALL be empty or a typed
:class:`V542InitMapError` is raised — never a bare RuntimeError, never a
loose fallback.

``dual_write`` entries are ``(source_key, target_key)`` pairs: an
ADDITIONAL copy of the source tensor into a target slot beyond
``mapping`` (loader-table orientation: source first).  Each pair is
declared exactly once; a dual target must not also be a mapping target.

Discipline: NOTHING here implements A1–A5; this is shared scaffolding
only, and no code in this package calls torch.compile (ADR-0033 §2.1).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator, Mapping, Sequence

import torch
import torch.nn as nn

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    EDGE_INDEX_KEY,
    GRAPH_CV_DELTA_KEY,
)
from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3
from pes2ts_core.pesn.models.key_atom_heads_v3 import ReactionProcessHeadV53MP
from pes2ts_core.pesn.models.v542.decoder import DirectMoveDecoder
from pes2ts_core.pesn.models.v542.interfaces import (
    N_PROCESS_PARAMS_V542,
    V542BackboneOutputError,
    V542InitError,
    V542InterfaceError,
    BackboneOutput,
    assert_no_torch_compile,
    validate_backbone_output,
)

#: Frozen R3 warm-start checkpoint path (config
#: ``training_contract.backbone_init.path``; the ONLY legal warm-start).
R3_CHECKPOINT_PATH: Final[str] = (
    "outputs/pesn/30h_campaign/r3_runs/5372d9b12807c3f0/checkpoint_best.pt")
#: Frozen R3 checkpoint byte identity (config ``backbone_init.sha256``).
R3_CHECKPOINT_SHA256: Final[str] = (
    "a2e46b6ad65cc18996d6c1660c5ae24c723397ca22526e52fa34e430afd0417f")

#: Frozen seed-recipe template (config ``init_policy.candidate_fresh_init``).
INIT_SEED_TEMPLATE: Final[str] = "v542-init|{run_seed}|{init_source_id}"
#: Frozen attribution overrides: diagnostic variants reuse their parent's
#: init stream (config ``init_source_id_overrides``).
INIT_SOURCE_ID_OVERRIDES: Final[dict[str, str]] = {"A1-C": "A1", "A5-C": "A5"}

#: Seed modulus of the frozen recipe (``% 2147483647``).
_INIT_SEED_MODULUS: Final[int] = 2_147_483_647
#: Frozen R3 backbone-only layout: ``payload["model"]`` key count.
_R3_BACKBONE_KEYS: Final[int] = 37


# ---------------------------------------------------------------------------
# typed errors (repo convention: every class carries ``code``/``detail``)
# ---------------------------------------------------------------------------
class V542InitMapError(V542InterfaceError):
    """Strict R3 semantic-mapping audit refusal
    (missing/unexpected/shape/dtype/double-consumption/value)."""


class V542InitSeedError(V542InterfaceError):
    """Frozen seed-recipe / construction-seed refusal."""


class V542CandidateBudgetError(V542InterfaceError):
    """Candidate active-parameter budget refusal (70k hard cap or an
    exact ``expected_active`` mismatch)."""


# ---------------------------------------------------------------------------
# frozen seed recipe (ADR-0033 §2.3; config init_policy.candidate_fresh_init)
# ---------------------------------------------------------------------------
def _require_seed(value: object, name: str) -> int:
    """Typed refusal for anything that is not a non-negative ``int``."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise V542InitSeedError(
            "V542_INIT_SEED_TYPE",
            f"{name} must be a non-negative int, got "
            f"{value!r} ({type(value).__name__})")
    if int(value) < 0:
        raise V542InitSeedError(
            "V542_INIT_SEED_TYPE",
            f"{name} must be non-negative, got {value!r}")
    return int(value)


def seed_init(run_seed: int, init_source_id: str) -> int:
    """Frozen fresh-init seed: ``seed_init(run_seed, init_source_id)``.

    Recipe (config ``candidate_fresh_init.seed_recipe``)::

        int(sha256(f"v542-init|{run_seed}|{init_source_id}").digest()[:8],
            "big") % 2147483647

    which is byte-identical to the frozen
    ``sha256_hexdigest[:16]_as_big_endian_int_mod_2147483647`` algorithm
    (``digest()[:8]`` IS ``hexdigest()[:16]`` — asserted in the test
    suite).  The result is always in ``[0, 2147483646]``.

    ``init_source_id`` is the RESOLVED id: run
    :func:`resolve_init_source_id` first for A1-C/A5-C so those
    diagnostics share their parent candidate's init stream.

    Raises :class:`V542InitSeedError` (``V542_INIT_SEED_TYPE``) on a
    non-int / negative / bool ``run_seed`` or a non-string / empty
    ``init_source_id``.
    """
    seed = _require_seed(run_seed, "run_seed")
    if not isinstance(init_source_id, str) or not init_source_id:
        raise V542InitSeedError(
            "V542_INIT_SEED_TYPE",
            "init_source_id must be a non-empty str, got "
            f"{init_source_id!r} ({type(init_source_id).__name__})")
    message = INIT_SEED_TEMPLATE.format(
        run_seed=seed, init_source_id=init_source_id)
    digest = hashlib.sha256(message.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % _INIT_SEED_MODULUS


def resolve_init_source_id(candidate_id: str) -> str:
    """``init_source_id`` of a candidate (default: the candidate id itself).

    Applies the frozen attribution overrides
    :data:`INIT_SOURCE_ID_OVERRIDES` — A1-C uses A1's init stream and
    A5-C uses A5's (config ``init_source_id_overrides``; ADR-0033 §2.3:
    attribution controls must not introduce an initialization nuisance).
    Raises :class:`V542InitSeedError` (``V542_INIT_SOURCE_ID``) on a
    non-string / empty ``candidate_id``.
    """
    if not isinstance(candidate_id, str) or not candidate_id:
        raise V542InitSeedError(
            "V542_INIT_SOURCE_ID",
            "candidate_id must be a non-empty str, got "
            f"{candidate_id!r} ({type(candidate_id).__name__})")
    return INIT_SOURCE_ID_OVERRIDES.get(candidate_id, candidate_id)


# ---------------------------------------------------------------------------
# frozen R3 warm-start loader (config backbone_init; ADR-0033 §2.3)
# ---------------------------------------------------------------------------
def load_r3_backbone_state(
        path: str = R3_CHECKPOINT_PATH,
        expected_sha256: str = R3_CHECKPOINT_SHA256,
) -> dict[str, torch.Tensor]:
    """Load the pinned R3 backbone state dict with byte-identity proof.

    SHA-256 is computed over the FILE BYTES and compared to
    ``expected_sha256`` BEFORE the file is opened by ``torch.load``.
    Typed refusals (all :class:`V542InitError`, all V5.4.2 construction
    taxonomy — the three new init errors do not cover byte identity):

    * ``V542_R3_INIT_MISSING`` — path absent / not a regular file;
    * ``V542_R3_INIT_SHA256`` — digest drift against the pinned value;
    * ``V542_R3_INIT_PAYLOAD`` — un-loadable payload, no ``model``
      mapping, a non-tensor state entry, or a key count other than the
      frozen 37 backbone keys.

    Returns ``payload["model"]`` as a plain ``dict[str, torch.Tensor]``
    (the R3 backbone-only layout: 37 keys, unprefixed ``StarEGNNV3``
    names).  No caching: every call re-verifies the bytes.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise V542InitError(
            "V542_R3_INIT_MISSING",
            f"R3 checkpoint not found at {path!r} — the frozen R3 "
            "warm-start source must exist byte-for-byte")
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    actual_sha256 = hasher.hexdigest()
    if actual_sha256 != expected_sha256:
        raise V542InitError(
            "V542_R3_INIT_SHA256",
            f"R3 checkpoint byte drift at {path!r}: sha256 "
            f"{actual_sha256} != expected {expected_sha256}")
    try:
        payload = torch.load(str(file_path), map_location="cpu",
                             weights_only=False)
    except Exception as exc:  # pragma: no cover - pinned bytes always load
        raise V542InitError(
            "V542_R3_INIT_PAYLOAD",
            f"cannot load R3 checkpoint {path!r}: {exc}") from exc
    if (not isinstance(payload, Mapping)
            or not isinstance(payload.get("model"), Mapping)):
        raise V542InitError(
            "V542_R3_INIT_PAYLOAD",
            "R3 checkpoint payload must contain a 'model' mapping "
            f"(got {type(payload).__name__})")
    model_state = payload["model"]
    for key, value in model_state.items():
        if not isinstance(value, torch.Tensor):
            raise V542InitError(
                "V542_R3_INIT_PAYLOAD",
                f"R3 state entry {key!r} is {type(value).__name__}, "
                "expected a tensor")
    if len(model_state) != _R3_BACKBONE_KEYS:
        raise V542InitError(
            "V542_R3_INIT_PAYLOAD",
            f"R3 backbone state has {len(model_state)} keys, the frozen "
            f"layout is {_R3_BACKBONE_KEYS}")
    return dict(model_state)


# ---------------------------------------------------------------------------
# frozen construction protocol (config init_policy.construction_order 1→6)
# ---------------------------------------------------------------------------
#: The CANONICAL frozen construction protocol — the exact six-step order
#: of config ``training_contract.init_policy.construction_order``.  This
#: tuple is what every :class:`BuiltCandidate` records as
#: ``construction_order``: candidate builders never fork, reorder,
#: lengthen or shorten it (per-candidate status lives in
#: ``executed_steps`` / ``deferred_steps``, never in this tuple).
CONSTRUCTION_ORDER: Final[tuple[str, ...]] = (
    "1_fresh_backbone_modules_seeded_by_seed_init",
    "2_load_R3_mapped_tensors_over_fresh_init_per_init_mapping",
    "3_zero_init_special_parameters_per_candidate_spec",
    "4_direct_move_decoder_with_shared_w_v_from_R3",
    "5_process_head_from_deterministic_anchor_replay",
    "6_set_seed(run_seed)_before_training",
)

#: Individual canonical step names (indexes into
#: :data:`CONSTRUCTION_ORDER`) — builders cite these constants so a
#: renamed config step is a compile-time-visible break, not silent
#: string drift.
STEP_FRESH_MODULES: Final[str] = CONSTRUCTION_ORDER[0]
STEP_R3_MAPPING: Final[str] = CONSTRUCTION_ORDER[1]
STEP_ZERO_INIT: Final[str] = CONSTRUCTION_ORDER[2]
STEP_SHARED_W_V: Final[str] = CONSTRUCTION_ORDER[3]
STEP_PROCESS_HEAD: Final[str] = CONSTRUCTION_ORDER[4]
STEP_TRAINING_SEED: Final[str] = CONSTRUCTION_ORDER[5]

#: The protocol steps EVERY builder defers: step 5 (the process head is
#: assembled by P3 via :func:`build_process_head`, not by the backbone
#: lane) and step 6 (the caller MUST run :func:`finish_training_seed`
#: before training).  A builder's ``deferred_steps`` is exactly this
#: tuple; the steps it DID run are its ``executed_steps`` (a strict
#: ordered subset of :data:`CONSTRUCTION_ORDER`).
BUILDER_DELEGATED_STEPS: Final[tuple[str, ...]] = (
    STEP_PROCESS_HEAD,
    STEP_TRAINING_SEED,
)


@contextlib.contextmanager
def construction_scope(run_seed: int, init_source_id: str) -> Iterator[int]:
    """Isolate candidate construction on the frozen ``seed_init`` stream.

    Wraps the caller's build in ``torch.random.fork_rng(devices=[])`` and
    seeds it with ``seed_init(run_seed, init_source_id)``; the caller's
    global torch CPU RNG stream is restored byte-identically on exit —
    on normal exit AND on exception (config ``construction_rng``: the
    build may never perturb the sampler/loss stream).  Yields the init
    seed actually used (run-identity record).

    Frozen construction order executed by the caller INSIDE the scope:

    1. fresh backbone modules built under ``seed_init``;
    2. R3 tensors mapped over the fresh init — :func:`apply_r3_mapping`;
    3. special parameters zero-initialized — :func:`zero_init_`;
    4. ``DirectMoveDecoder`` with the SHARED ``w_v`` from R3 —
       :func:`apply_r3_w_v` (or :func:`apply_shared_w_v` when the
       backbone adopts the decoder's ``w_v`` object);
    5. ``ReactionProcessHeadV53MP`` from the deterministic anchor —
       :func:`build_process_head`;
    6. AFTER leaving the scope the caller MUST call
       :func:`finish_training_seed` (with the run seed) before training.
    """
    seed = seed_init(run_seed, init_source_id)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield seed


def finish_training_seed(run_seed: int) -> None:
    """Step 6: reset the global torch RNG to ``run_seed`` after the build.

    Must be called AFTER :func:`construction_scope` has been left, so
    training (sampler, loss, dropout-free but stochastic ops) starts from
    the frozen run seed.  ``torch.manual_seed`` runs, then the reset is
    asserted via ``torch.initial_seed()`` — a seed torch silently
    remapped (e.g. one outside the range torch preserves verbatim) is a
    typed :class:`V542InitSeedError` (``V542_INIT_SEED_RESET``), never a
    silent drift.
    """
    seed = _require_seed(run_seed, "run_seed")
    torch.manual_seed(seed)
    if int(torch.initial_seed()) != seed:
        raise V542InitSeedError(
            "V542_INIT_SEED_RESET",
            f"manual_seed({seed}) did not take: torch.initial_seed() is "
            f"{torch.initial_seed()} — the training seed was not reset")


def build_process_head(
        anchor_seed: int,
) -> ReactionProcessHeadV53MP:
    """Deterministic process-head anchor replay (config
    ``init_policy.process_head_anchor.method: fork_rng_replay``).

    Replays the frozen legacy construction ORDER under the anchor seed —
    ``manual_seed(anchor_seed)`` → ``StarEGNNV3()`` (the backbone draw
    that precedes the head in ``KeyAtomV53MPModel.__init__``, discarded)
    → ``ReactionProcessHeadV53MP()`` — inside
    ``torch.random.fork_rng(devices=[])`` so the call is RNG-NEUTRAL for
    the caller.  The head is excluded from ``w_v`` concerns by
    construction (``excludes_w_v: true``: ``w_v`` comes from R3 for every
    candidate).  The build asserts the frozen 2,178-parameter count.

    The anchor seeds are the frozen run seeds ``{17, 29, 43}``; the
    replay algorithm + seed go into ``process_head_anchor_sha256`` of the
    run identity (the binary artifact is optional).
    """
    seed = _require_seed(anchor_seed, "anchor_seed")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        _ = StarEGNNV3()                      # legacy draw order (discarded)
        head = ReactionProcessHeadV53MP()     # the anchor head
    n_params = sum(int(p.numel()) for p in head.parameters())
    if n_params != N_PROCESS_PARAMS_V542:
        raise V542InitError(
            "V542_ANCHOR_HEAD_PARAMS",
            f"anchored ReactionProcessHeadV53MP has {n_params} params, "
            f"frozen count is {N_PROCESS_PARAMS_V542}")
    return head


def apply_r3_w_v(
        move_decoder: DirectMoveDecoder,
        source: Mapping[str, torch.Tensor] | None = None,
) -> None:
    """Construction step 4 (decoder-only): load the R3 ``w_v`` (49
    params) into the SHARED decoder slot (config ``construction_order``
    item 4; ALL candidates — A1/A1-C/A2/A3/A4/A5/A5-C — start MOVE from
    the same SHA-verified R3 values).

    ``source=None`` loads the pinned R3 via :func:`load_r3_backbone_state`
    (the SHA-256 byte gate re-verifies on every call); an injected
    37-key ``source`` mapping (tests / a builder's already-resolved
    payload) is used verbatim — builders pass the SAME resolved source
    into both :func:`apply_r3_mapping` and this loader, so the backbone
    mapping and the decoder ``w_v`` can never come from different
    payloads.

    Typed refusals (all :class:`V542InitError`): ``V542_DECODER_TYPE``
    (non-``DirectMoveDecoder``), ``V542_WV_SOURCE`` (non-mapping source
    or missing ``w_v.weight``/``w_v.bias`` tensors), ``V542_WV_MISMATCH``
    (shape/dtype drift, checked BEFORE any copy — no heuristics),
    ``V542_WV_NOT_EQUAL`` (post-copy ``torch.equal`` failed).  Returns
    nothing; on success ``decoder.w_v`` is bitwise-equal to the source
    tensors.
    """
    if not isinstance(move_decoder, DirectMoveDecoder):
        raise V542InitError(
            "V542_DECODER_TYPE",
            "apply_r3_w_v requires the shared DirectMoveDecoder "
            f"(got {type(move_decoder).__name__})")
    resolved: Mapping[str, torch.Tensor]
    if source is None:
        resolved = load_r3_backbone_state()
    elif isinstance(source, Mapping):
        resolved = source
    else:
        raise V542InitError(
            "V542_WV_SOURCE",
            "injected w_v source must be a mapping of tensors, got "
            f"{type(source).__name__}")
    src_weight = resolved.get("w_v.weight")
    src_bias = resolved.get("w_v.bias")
    if not isinstance(src_weight, torch.Tensor) or \
            not isinstance(src_bias, torch.Tensor):
        raise V542InitError(
            "V542_WV_SOURCE",
            "source payload lacks w_v.weight/w_v.bias tensors")
    weight, bias = move_decoder.w_v.weight, move_decoder.w_v.bias
    if (tuple(weight.shape) != tuple(src_weight.shape)
            or tuple(bias.shape) != tuple(src_bias.shape)
            or weight.dtype != src_weight.dtype
            or bias.dtype != src_bias.dtype):
        raise V542InitError(
            "V542_WV_MISMATCH",
            "decoder w_v slot does not match the source w_v: module "
            f"{tuple(weight.shape)}/{weight.dtype} vs source "
            f"{tuple(src_weight.shape)}/{src_weight.dtype}")
    with torch.no_grad():
        weight.copy_(src_weight)
        bias.copy_(src_bias)
    if not (torch.equal(move_decoder.w_v.weight, src_weight)
            and torch.equal(move_decoder.w_v.bias, src_bias)):
        raise V542InitError(  # pragma: no cover - copy_ cannot diverge
            "V542_WV_NOT_EQUAL",
            "post-copy torch.equal FAILED for the decoder w_v")


def apply_shared_w_v(backbone: nn.Module, move_decoder: DirectMoveDecoder,
                     source: Mapping[str, torch.Tensor] | None = None,
                     ) -> None:
    """Construction step 4: put the R3 ``w_v`` (49 params) into the
    SHARED decoder slot (config ``construction_order`` item 4; all
    candidates share the same ``w_v`` values).

    First asserts the shared-wiring invariant
    ``backbone.w_v is move_decoder.w_v`` (typed :class:`V542InitError`
    ``V542_WV_NOT_SHARED`` — a forked ``w_v`` would double-update in the
    optimizer and break the joint invariant), then DELEGATES the actual
    load to :func:`apply_r3_w_v` (SHA-pinned R3 by default, or the same
    injected ``source`` the builder used for
    :func:`apply_r3_mapping`), which shape/dtype-checks, copies and
    asserts ``torch.equal`` on both tensors.  The backbone alias shares
    the decoder's module object, so the post-copy equality holds for
    ``backbone.w_v`` by identity.
    """
    if not isinstance(move_decoder, DirectMoveDecoder):
        raise V542InitError(
            "V542_DECODER_TYPE",
            "apply_shared_w_v requires the shared DirectMoveDecoder "
            f"(got {type(move_decoder).__name__})")
    if not isinstance(backbone, nn.Module) or not hasattr(backbone, "w_v"):
        raise V542InitError(
            "V542_WV_NOT_SHARED",
            "backbone must expose the shared w_v module "
            f"(got {type(backbone).__name__})")
    if backbone.w_v is not move_decoder.w_v:
        raise V542InitError(
            "V542_WV_NOT_SHARED",
            "backbone.w_v is not the decoder's w_v module object — the "
            "shared-w_v invariant (backbone.w_v is move_decoder.w_v) must "
            "hold before the R3 w_v can be applied")
    apply_r3_w_v(move_decoder, source)


# ---------------------------------------------------------------------------
# strict R3 semantic mapping audit (config init_policy.mapping/audit)
# ---------------------------------------------------------------------------
def _mapping_table_digest(table: Mapping[str, str]) -> str:
    """LOCAL placeholder digest of the effective ``{target: source}``
    table (mapping + declared dual-write), loader-style.  The canonical
    SHA recipe for mapping tables is a separate advisory lane; this
    records a clearly-labelled local digest of the table only."""
    payload = json.dumps(dict(sorted(table.items())),
                         sort_keys=True, separators=(",", ":"))
    return "LOCAL_PLACEHOLDER:" + hashlib.sha256(
        payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MappingAudit:
    """Auditable record of one :func:`apply_r3_mapping` run.

    * ``mapping_table_sha256`` — ``LOCAL_PLACEHOLDER:``-prefixed local
      digest of the effective ``{target: source}`` table;
    * ``source_keys`` — every audited payload key (sorted);
    * ``target_keys`` — every module key written (sorted);
    * ``dual_write_aliases`` — the declared ``(source_key, target_key)``
      additional writes;
    * ``torch_equal_verified`` — targets proven ``torch.equal`` to their
      source AFTER the copy;
    * ``missing`` / ``unexpected`` / ``shape_mismatch`` /
      ``dtype_mismatch`` — always EMPTY in a returned audit (any
      non-empty category raises :class:`V542InitMapError` before it can
      be returned).
    """

    mapping_table_sha256: str
    source_keys: tuple[str, ...]
    target_keys: tuple[str, ...]
    dual_write_aliases: tuple[tuple[str, str], ...]
    torch_equal_verified: tuple[str, ...]
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    shape_mismatch: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] = ()
    dtype_mismatch: tuple[tuple[str, str, str], ...] = ()


def apply_r3_mapping(
        module: nn.Module,
        source_state: Mapping[str, torch.Tensor],
        mapping: Mapping[str, str],
        *,
        consumed_source_keys: Sequence[str] | None = None,
        dual_write: Sequence[tuple[str, str]] = (),
) -> MappingAudit:
    """Strict explicit mapping of R3 tensors onto a candidate module.

    Args:
        module: target module (its ``state_dict()`` supplies the target
            slots; slots outside ``mapping`` stay untouched — they are
            the candidate's fresh parameters).
        source_state: the SHA-verified R3 payload
            (:func:`load_r3_backbone_state` returns the 37-key layout).
        mapping: ``{target_key: source_key}`` — the pre-registered
            semantic table (config ``init_policy.mapping``:
            ``explicit_semantic_pre_registered_no_runtime_shape_
            heuristics``).
        consumed_source_keys: payload keys consumed OUTSIDE this call —
            e.g. ``w_v.*`` by :func:`apply_r3_w_v` /
            :func:`apply_shared_w_v` (construction step 4 runs after
            this audit, on the same resolved source), or payload keys
            a candidate deliberately does not map.  Every payload key
            must be accounted exactly once across mapping + dual-write +
            this list (strict reading of ``mapping_audit``: "every
            source key consumed exactly once ... no empty slots").
        dual_write: additional ``(source_key, target_key)`` copies that
            declare the legal double consumption of a source (the
            ``w_v`` alias pattern).  Each pair declared exactly once; a
            dual target must not also be a mapping target.

    Audit order (typed :class:`V542InitMapError`, code in parentheses):
    ``missing`` (``V542_INIT_MAP_MISSING``) → ``unexpected``
    (``V542_INIT_MAP_UNEXPECTED``) → undeclared double consumption /
    duplicate dual declaration (``V542_INIT_MAP_DOUBLE_CONSUME``) →
    target written from two sources (``V542_INIT_MAP_TARGET_CONFLICT``)
    → ``shape_mismatch`` (``V542_INIT_MAP_SHAPE``) → ``dtype_mismatch``
    (``V542_INIT_MAP_DTYPE``).  ONLY after every category is empty does
    the copy run (``copy_`` per audited pair — no shape heuristics, no
    ``copy_`` on mismatch), followed by per-key ``torch.equal``
    verification (``V542_INIT_MAP_VALUE``).

    Returns the frozen :class:`MappingAudit` record (all four loader
    categories empty by construction).
    """
    if not isinstance(module, nn.Module):
        raise V542InitMapError(
            "V542_INIT_MAP_TARGET",
            f"module must be an nn.Module, got {type(module).__name__}")
    if not isinstance(source_state, Mapping):
        raise V542InitMapError(
            "V542_INIT_MAP_SOURCE",
            f"source_state must be a mapping, got {type(source_state).__name__}")
    if not isinstance(mapping, Mapping):
        raise V542InitMapError(
            "V542_INIT_MAP_TABLE",
            f"mapping must be a mapping, got {type(mapping).__name__}")
    if not all(isinstance(k, str) and isinstance(v, str)
               for k, v in mapping.items()):
        raise V542InitMapError(
            "V542_INIT_MAP_TABLE",
            "mapping must be {target_key: source_key} of plain strings")
    for key, value in source_state.items():
        if not isinstance(value, torch.Tensor):
            raise V542InitMapError(
                "V542_INIT_MAP_SOURCE",
                f"source entry {key!r} is {type(value).__name__}, "
                "expected a tensor")
    try:
        pairs = [(str(src), str(tgt)) for src, tgt in dual_write]
    except (TypeError, ValueError) as exc:
        raise V542InitMapError(
            "V542_INIT_MAP_DUAL",
            "dual_write entries must be (source_key, target_key) pairs "
            f"({exc})") from exc
    declared: list[str] = []
    for entry in (consumed_source_keys or ()):
        if not isinstance(entry, str):
            raise V542InitMapError(
                "V542_INIT_MAP_CONSUMED",
                f"consumed_source_keys entries must be str, got "
                f"{entry!r} ({type(entry).__name__})")
        declared.append(entry)

    module_state = module.state_dict()
    mapping_count: Counter = Counter(mapping.values())
    pair_count: Counter = Counter(src for src, _ in pairs)
    consumed_count: Counter = Counter(declared)

    # -- missing: declared source absent from payload / target absent -------
    missing: set[str] = set()
    for src in list(mapping.values()) + [s for s, _ in pairs] + declared:
        if src not in source_state:
            missing.add(str(src))
    for tgt in list(mapping) + [t for _, t in pairs]:
        if tgt not in module_state:
            missing.add(str(tgt))

    # -- unexpected: payload keys nobody accounts for -----------------------
    unexpected = tuple(sorted(
        key for key in source_state
        if mapping_count[key] + pair_count[key] + consumed_count[key] == 0))

    # -- consumption: exactly once, except declared dual-write --------------
    double: list[str] = []
    for src in sorted(set(mapping_count) | set(pair_count) | set(consumed_count)):
        total = (mapping_count[src] + pair_count[src] + consumed_count[src])
        allowed = 1 + pair_count[src]
        if total > allowed:
            double.append(
                f"source {src!r} consumed {total}x (allowed {allowed}x)")
    if len(pairs) != len(set(pairs)):
        dup = sorted({p for p in pairs if pairs.count(p) > 1})
        double.append(f"duplicate dual_write declarations: {dup}")

    # -- each target written from exactly ONE source ------------------------
    writes: list[tuple[str, str]] = ([(tgt, src) for tgt, src in mapping.items()]
                                     + [(tgt, src) for src, tgt in pairs])
    target_source: dict[str, str] = {}
    conflicts: list[str] = []
    for tgt, src in writes:
        if tgt in target_source:
            conflicts.append(
                f"target {tgt!r} written by {target_source[tgt]!r} "
                f"and {src!r}")
        else:
            target_source[tgt] = src

    # -- shape / dtype audit (both sides present; NO heuristics) ------------
    shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    dtype_mismatch: list[tuple[str, str, str]] = []
    for tgt, src in writes:
        if src not in source_state or tgt not in module_state:
            continue  # already reported via missing
        ref, value = module_state[tgt], source_state[src]
        if tuple(ref.shape) != tuple(value.shape):
            shape_mismatch.append(
                (tgt, tuple(ref.shape), tuple(value.shape)))
        if ref.dtype != value.dtype:
            dtype_mismatch.append((tgt, str(ref.dtype), str(value.dtype)))

    if missing:
        raise V542InitMapError(
            "V542_INIT_MAP_MISSING",
            "strict audit: missing keys (source or target) — "
            + ", ".join(sorted(missing)))
    if unexpected:
        raise V542InitMapError(
            "V542_INIT_MAP_UNEXPECTED",
            "strict audit: source keys consumed by no declaration: "
            + ", ".join(unexpected))
    if double:
        raise V542InitMapError(
            "V542_INIT_MAP_DOUBLE_CONSUME",
            "strict audit: undeclared double consumption — " + "; ".join(double))
    if conflicts:
        raise V542InitMapError(
            "V542_INIT_MAP_TARGET_CONFLICT",
            "strict audit: target written from multiple sources — "
            + "; ".join(conflicts))
    if shape_mismatch:
        raise V542InitMapError(
            "V542_INIT_MAP_SHAPE",
            "strict audit: shape mismatch — "
            + ", ".join(f"{t}: module{ms} != source{ss}"
                        for t, ms, ss in shape_mismatch))
    if dtype_mismatch:
        raise V542InitMapError(
            "V542_INIT_MAP_DTYPE",
            "strict audit: dtype mismatch — "
            + ", ".join(f"{t}: module {md} != source {sd}"
                        for t, md, sd in dtype_mismatch))

    # -- copy only after EVERY category is clean ----------------------------
    with torch.no_grad():
        for tgt, src in writes:
            module_state[tgt].copy_(source_state[src])

    # -- per-key torch.equal verification -----------------------------------
    post = module.state_dict()
    verified: list[str] = []
    for tgt in sorted(target_source):
        src = target_source[tgt]
        if not torch.equal(post[tgt], source_state[src]):
            raise V542InitMapError(  # pragma: no cover - copy_ cannot diverge
                "V542_INIT_MAP_VALUE",
                f"post-copy torch.equal FAILED for {tgt!r}")
        verified.append(tgt)

    return MappingAudit(
        mapping_table_sha256=_mapping_table_digest(target_source),
        source_keys=tuple(sorted(source_state)),
        target_keys=tuple(sorted(target_source)),
        dual_write_aliases=tuple(pairs),
        torch_equal_verified=tuple(verified),
    )


# ---------------------------------------------------------------------------
# builder audit record (C3/P3): what a candidate build actually did
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BuiltCandidate:
    """Auditable record returned by every candidate build.

    * ``candidate_id`` — the frozen id the build resolved (``"A1"``,
      ``"A1-C"``, ``"A2"``, ``"A3"``, ``"A4"``, ``"A5"``, ``"A5-C"``);
    * ``backbone`` — the built backbone module;
    * ``move_decoder`` — the SHARED ``DirectMoveDecoder`` seam the build
      wired (``w_v`` R3-loaded, frozen ``a_ref``);
    * ``mapping_audit`` — the :class:`MappingAudit` returned by the
      ACTUAL :func:`apply_r3_mapping` call inside the build (not a
      recomputation);
    * ``seed_init`` — the construction seed the scope ran under
      (``seed_init(run_seed, init_source_id)``);
    * ``init_source_id`` — the resolved init-stream attribution;
    * ``construction_order`` — the CANONICAL frozen six-step protocol
      (config ``init_policy.construction_order``): always exactly
      :data:`CONSTRUCTION_ORDER`, identical for every candidate —
      candidate builders never record a builder-specific list here;
    * ``executed_steps`` — the exact subset of :data:`CONSTRUCTION_ORDER`
      THIS builder performed, mapped to the canonical step names, in
      protocol order (steps 1–4: fresh modules, R3 mapping, zero-init,
      shared ``w_v``; the budget gate is a contract check, NOT a
      protocol step and never appears in this list);
    * ``deferred_steps`` — the canonical steps this builder delegates
      (always :data:`BUILDER_DELEGATED_STEPS`: step 5 process head is
      assembled by P3, step 6 ``finish_training_seed`` is the caller's
      job).
    """

    candidate_id: str
    backbone: nn.Module
    move_decoder: DirectMoveDecoder
    mapping_audit: MappingAudit
    seed_init: int
    init_source_id: str
    executed_steps: tuple[str, ...]
    construction_order: tuple[str, ...] = CONSTRUCTION_ORDER
    deferred_steps: tuple[str, ...] = BUILDER_DELEGATED_STEPS


# ---------------------------------------------------------------------------
# per-candidate zero-init (config init_policy.per_candidate_zero_init)
# ---------------------------------------------------------------------------
def zero_init_(linear: nn.Linear) -> None:
    """Zero a special parameter slot IN PLACE (config
    ``per_candidate_zero_init``: A1/A1-C ``W_d``, A2 ``V_0``/final
    outputs, A3 ``W_o``, A4 output projection, A5 ``W_g``).

    ``weight`` and ``bias`` (when present) are set to exact zeros under
    ``torch.no_grad`` — the module object, dtype and shape are untouched,
    so a subsequent forward is structurally unaffected.  A non-``nn.Linear``
    is a typed :class:`V542InitError` (``V542_ZERO_INIT_TYPE``).
    """
    if not isinstance(linear, nn.Linear):
        raise V542InitError(
            "V542_ZERO_INIT_TYPE",
            "zero_init_ expects an nn.Linear slot, got "
            f"{type(linear).__name__}")
    with torch.no_grad():
        linear.weight.zero_()
        if linear.bias is not None:
            linear.bias.zero_()


# ---------------------------------------------------------------------------
# budget + backbone contract gates (config param_budget / interface)
# ---------------------------------------------------------------------------
def assert_candidate_budget(
        backbone: nn.Module,
        *,
        expected_active: int | None = None,
        max_active: int = 70_000,
) -> int:
    """Measure and assert the candidate's active-parameter count.

    Prefers the :class:`~pes2ts_core.pesn.models.v542.interfaces.Backbone`
    protocol property ``active_param_count`` (excludes any decoder-owned
    shared ``w_v``); falls back to ``sum(p.numel())`` for a raw module.
    Refusals (typed :class:`V542CandidateBudgetError`):

    * ``V542_CANDIDATE_BUDGET`` — measured count ``> max_active``
      (frozen hard cap 70,000, config ``param_budget.backbone_active_max``);
    * ``V542_CANDIDATE_ACTIVE_MISMATCH`` — ``expected_active`` given and
      the measured count differs (exact-assertion discipline: candidates
      assert their measured count, never an estimate).

    Returns the measured count.
    """
    if not isinstance(backbone, nn.Module):
        raise V542CandidateBudgetError(
            "V542_CANDIDATE_BUDGET",
            f"backbone must be an nn.Module, got {type(backbone).__name__}")
    if expected_active is not None and not isinstance(expected_active, int):
        raise V542CandidateBudgetError(
            "V542_CANDIDATE_ACTIVE_MISMATCH",
            f"expected_active must be an int, got "
            f"{expected_active!r} ({type(expected_active).__name__})")
    if not isinstance(max_active, int) or isinstance(max_active, bool):
        raise V542CandidateBudgetError(
            "V542_CANDIDATE_BUDGET",
            f"max_active must be an int, got {max_active!r}")
    measured: object
    try:
        measured = backbone.active_param_count
    except AttributeError:
        measured = sum(int(p.numel()) for p in backbone.parameters())
    if callable(measured):  # tolerate a method-style accessor
        measured = measured()
    count = int(measured)
    if count > int(max_active):
        raise V542CandidateBudgetError(
            "V542_CANDIDATE_BUDGET",
            f"active params {count} > max_active {int(max_active)} "
            "(backbone_active_max is a hard construction-time cap)")
    if expected_active is not None and count != int(expected_active):
        raise V542CandidateBudgetError(
            "V542_CANDIDATE_ACTIVE_MISMATCH",
            f"active params {count} != expected_active "
            f"{int(expected_active)} — candidates assert their MEASURED "
            "count, estimates are never accepted")
    return count


def check_backbone_contract(
        backbone: nn.Module,
        batch: Mapping[str, torch.Tensor],
) -> None:
    """Run the frozen backbone contract on one real forward.

    Checks, in order:

    1. no torch.compile-wrapped module anywhere in the tree
       (``V542_TORCH_COMPILE_FORBIDDEN``, ADR-0033 §2.1 — checked BEFORE
       the forward, so a compiled tree never even runs);
    2. ``backbone(batch)`` returns a :class:`BackboneOutput` with
       ``z [G, 64]`` (``G = batch["cv_target_delta"].rows``) and
       ``edge_msg [E, 48]`` (``E = batch["edge_index"].columns``), fp32
       and FINITE — via the frozen
       :func:`validate_backbone_output` (typed
       ``V542_BACKBONE_Z_SHAPE`` / ``V542_BACKBONE_EDGE_SHAPE`` /
       ``V542_BACKBONE_DTYPE`` / ``V542_BACKBONE_NONFINITE``);
    3. a LIVE autograd graph: both ``z`` and ``edge_msg`` must carry
       ``requires_grad`` + ``grad_fn`` (``V542_BACKBONE_NO_GRAD``) —
       a detached/no_grad build cannot train and is refused.
    """
    if not isinstance(backbone, nn.Module):
        raise V542BackboneOutputError(
            "V542_BACKBONE_CONTRACT",
            f"backbone must be an nn.Module, got {type(backbone).__name__}")
    if not isinstance(batch, Mapping):
        raise V542BackboneOutputError(
            "V542_BACKBONE_CONTRACT",
            f"batch must be a mapping, got {type(batch).__name__}")
    assert_no_torch_compile(backbone)
    try:
        n_graphs = int(batch[GRAPH_CV_DELTA_KEY].shape[0])
        n_edges = int(batch[EDGE_INDEX_KEY].shape[1])
    except (KeyError, TypeError, AttributeError) as exc:
        raise V542BackboneOutputError(
            "V542_BACKBONE_CONTRACT",
            f"batch lacks {GRAPH_CV_DELTA_KEY}/{EDGE_INDEX_KEY} shapes "
            f"needed by the contract check ({exc})") from exc
    output = backbone(batch)
    validate_backbone_output(output, n_graphs=n_graphs, n_edges=n_edges)
    for name, tensor in (("z", output.z), ("edge_msg", output.edge_msg)):
        if not tensor.requires_grad or tensor.grad_fn is None:
            raise V542BackboneOutputError(
                "V542_BACKBONE_NO_GRAD",
                f"{name} must carry a live autograd graph "
                "(requires_grad + grad_fn) — a detached or no_grad build "
                "cannot be trained (config edge_msg.autograd: true)")


__all__ = [
    "BUILDER_DELEGATED_STEPS",
    "CONSTRUCTION_ORDER",
    "INIT_SEED_TEMPLATE",
    "INIT_SOURCE_ID_OVERRIDES",
    "R3_CHECKPOINT_PATH",
    "R3_CHECKPOINT_SHA256",
    "STEP_FRESH_MODULES",
    "STEP_PROCESS_HEAD",
    "STEP_R3_MAPPING",
    "STEP_SHARED_W_V",
    "STEP_TRAINING_SEED",
    "STEP_ZERO_INIT",
    "BuiltCandidate",
    "MappingAudit",
    "V542CandidateBudgetError",
    "V542InitMapError",
    "V542InitSeedError",
    "apply_r3_mapping",
    "apply_r3_w_v",
    "apply_shared_w_v",
    "assert_candidate_budget",
    "build_process_head",
    "check_backbone_contract",
    "construction_scope",
    "finish_training_seed",
    "load_r3_backbone_state",
    "resolve_init_source_id",
    "seed_init",
    "zero_init_",
]
