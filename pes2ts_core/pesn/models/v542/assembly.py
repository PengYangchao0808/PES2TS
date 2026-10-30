"""V5.4.2 candidate-agnostic full-model assembly (WP3 lane P3-B;
ADR-0033 §2.1/§2.3/§2.6; config ``training_contract.init_policy``).

One entry point — :func:`assemble_v542_model` — turns ANY candidate id
(``A0`` the matched baseline, ``A1``–``A5`` and the ``A1-C``/``A5-C``
diagnostics) into a TRAINABLE full model behind the frozen external
surface::

    batch ─► backbone ─► BackboneOutput(z, edge_msg)
                            │                │
                            ▼                ▼
                  process_head (z)    move_decoder (edge_msg, …)
                    q/endpoint/order         u

What this module FREEZES (nothing here modifies an existing V5.4.2
file — the WP1 ``V542Model`` forward and the WP2 candidate registry are
imported and REUSED):

* **the full-model wrapper** — :class:`AssembledV542Module` SUBCLASSES
  :class:`~pes2ts_core.pesn.models.v542.model.V542Model` and inherits
  its frozen ``forward`` (exactly the five keys ``{u, z, q,
  endpoint_logit, order_score}``), ``forward_move`` (``{u, z}``) and the
  BY-``id()`` :meth:`~V542Model.param_groups_by_id` classifier with all
  of its typed refusals.  Only the CONSTRUCTOR is re-expressed: the WP1
  constructor asserts the A0 migration invariants, so the
  candidate-agnostic constructor runs ``nn.Module.__init__`` directly
  and re-expresses those invariants PER CANDIDATE (same error codes,
  same budgets).  ``order_score IS q`` comes from the reused
  :class:`~pes2ts_core.pesn.models.key_atom_heads_v3.
  ReactionProcessHeadV53MP` (never forked, 2,178 params);
* **the frozen construction order** (config
  ``init_policy.construction_order`` 1→6): the whole assembly runs in
  ONE ``torch.random.fork_rng(devices=[])`` scope — fresh decoder, the
  candidate build (:func:`~pes2ts_core.pesn.models.v542.candidates.
  build_candidate` for A1–A5/A1-C/A5-C, the equivalent local A0 record
  below for the baseline), then the anchored process head (step 5) —
  and AFTER the scope :func:`~pes2ts_core.pesn.models.v542.candidates.
  finish_training_seed` (step 6) resets the global torch RNG to the run
  seed and the reset is VERIFIED, never assumed;
* **the anchor seed rule** — the process head is replayed with
  ``anchor_seed IS run_seed``, and the run seed must be one of the
  frozen ``{17, 29, 43}`` (config ``process_head_anchor.seeds``); any
  other seed is a typed refusal before anything is built;
* **exact counts** — the candidate's ACTIVE count is asserted against
  the frozen registry value (``CANDIDATE_PARAM_COUNTS``; A0 =
  ``A0_ACTIVE_PARAMS_V542``), ``head == 2,178``,
  ``core == active + 49``, ``joint == active + 49 + 2,178`` and the
  hard cap ``joint <= 72,227`` (config ``param_budget.joint_max``);
* **param groups BY ``id()``** — ``core = backbone ∪ move_decoder`` at
  ``lr 1e-4`` and ``head = process_head`` at ``lr 3e-4``; every unique
  parameter appears EXACTLY once (the shared ``w_v`` is one tensor, so
  the optimizer can never double-update it), no duplicate ids;
* **identity recording** — whatever identity dict the caller passes is
  recorded verbatim (plus its canonical SHA-256); a dirty or absent
  identity is refused ONLY when ``require_identity=True``.

Discipline: no ``torch.compile`` anywhere (ADR-0033 §2.1, checked at
construction), no new parameters, no modification of legacy/WP1/WP2
files, no loss/sampler/threshold code here (those live in the frozen
trainers this package's trainer reuses).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Final, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from pes2ts_core.pesn.models.key_atom_heads_v3 import ReactionProcessHeadV53MP
from pes2ts_core.pesn.models.v542.backbone_a0 import A0LegacyBackbone
from pes2ts_core.pesn.models.v542.candidates import (
    CANDIDATE_BUILDERS,
    CANDIDATE_PARAM_COUNTS,
    CONSTRUCTION_ORDER,
    BuiltCandidate,
    V542InitMapError,
    V542InitSeedError,
    apply_r3_mapping,
    apply_shared_w_v,
    assert_candidate_budget,
    build_candidate,
    build_process_head,
    construction_scope,
    finish_training_seed,
    load_r3_backbone_state,
)
from pes2ts_core.pesn.models.v542.decoder import (
    DirectMoveDecoder,
    assert_decoder_a_ref_frozen,
)
from pes2ts_core.pesn.models.v542.identity import canonical_sha256
from pes2ts_core.pesn.models.v542.interfaces import (
    A0_ACTIVE_PARAMS_V542,
    A0_JOINT_PARAMS_V542,
    N_DECODER_PARAMS_V542,
    N_PROCESS_PARAMS_V542,
    V542InitError,
    assert_no_torch_compile,
)
from pes2ts_core.pesn.models.v542.model import V542Model

__all__ = [
    "ANCHOR_SEEDS_V542",
    "A0",
    "JOINT_PARAM_MAX_V542",
    "LR_CORE_V542",
    "LR_HEAD_V542",
    "V542_CANDIDATE_IDS_V542",
    "AssembledV542Model",
    "AssembledV542Module",
    "V542AssemblyError",
    "assemble_v542_model",
    "identity_sha256",
]

#: The A0 matched-baseline id (the registry ``build_candidate`` whitelist
#: covers A1–A5 + A1-C/A5-C only; A0 is assembled by this module).
A0: Final[str] = "A0"

#: Frozen process-head anchor seeds (config
#: ``init_policy.process_head_anchor.seeds``): the anchor seed IS the run
#: seed, and only these three run seeds exist in the campaign.
ANCHOR_SEEDS_V542: Final[Tuple[int, ...]] = (17, 29, 43)

#: Frozen param-group lrs (config ``init_policy.param_groups``).
LR_CORE_V542: Final[float] = 1.0e-4     # core = backbone ∪ move_decoder
LR_HEAD_V542: Final[float] = 3.0e-4     # head = process_head

#: Frozen joint-parameter hard cap (config ``param_budget.joint_max``).
JOINT_PARAM_MAX_V542: Final[int] = 72_227

#: Every candidate id this assembler accepts (registry seven + A0).
V542_CANDIDATE_IDS_V542: Final[Tuple[str, ...]] = (
    (A0,) + tuple(CANDIDATE_BUILDERS))

#: The frozen A0 ``w_v`` source keys — consumed by construction step 4
#: (:func:`apply_shared_w_v`), declared up front in the A0 mapping audit.
_A0_WV_SOURCE_KEYS: Final[Tuple[str, ...]] = ("w_v.weight", "w_v.bias")

#: The frozen A0 backbone-only key layout (config ``init_mapping``: A0 =
#: all 37 R3 backbone keys identity-mapped, ``w_v`` via the shared slot).
_A0_BACKBONE_KEYS: Final[int] = 37


class V542AssemblyError(V542InitError):
    """Typed refusal of the assembly seam (identity / counts / device)."""


# ---------------------------------------------------------------------------
# identity recording (config ``identity_contract``; the SHA KEY SET is the
# runner lane's job — this seam only proves presence/cleanliness on demand)
# ---------------------------------------------------------------------------
def identity_sha256(identity: Mapping[str, Any]) -> str:
    """Canonical JSON SHA-256 of an identity block — THE strict recipe.

    Delegates to :func:`pes2ts_core.pesn.models.v542.identity.
    canonical_sha256` (gate-3 C1: ONE canonicalization everywhere).  A
    non-JSON-serializable identity value is a TYPED refusal
    (``V542_IDENTITY_SERIALIZE``) — never a silent ``str()`` fallback, so
    the model-side digest and the trainer-side digest can never diverge
    in algorithm AND can never silently stringify a foreign value.
    """
    return canonical_sha256(dict(identity))


def _identity_issues(identity: Any) -> list:
    """Cleanliness problems of a candidate identity dict (empty = clean)."""
    if not isinstance(identity, Mapping):
        return ["identity is not a Mapping"]
    issues: list = []
    for key, value in dict(identity).items():
        if not isinstance(key, str) or not key:
            issues.append(f"non-str/empty identity key {key!r}")
        elif value is None:
            issues.append(f"identity[{key!r}] is None (absent SHA)")
        elif isinstance(value, str) and not value:
            issues.append(f"identity[{key!r}] is an empty string")
        elif not isinstance(value, (str, int, float, bool, Mapping, list,
                                    tuple)):
            issues.append(
                f"identity[{key!r}] is a {type(value).__name__}, "
                "expected a scalar SHA/value or a nested mapping")
    return issues


def _resolve_identity(identity: Optional[Mapping[str, Any]],
                      *, require_identity: bool) -> Dict[str, Any]:
    """Record whatever identity dict the caller passed — dirty/absent is
    refused ONLY under ``require_identity=True`` (spec: the runner lane
    opts into the strict check; a fixture lane may record ``{}``)."""
    if not isinstance(identity, Mapping):
        if require_identity:
            raise V542AssemblyError(
                "V542_IDENTITY_ABSENT",
                "require_identity=True but no identity mapping was passed "
                f"(got {identity!r})")
        if identity is None:
            return {}
        raise V542AssemblyError(
            "V542_IDENTITY_TYPE",
            f"identity must be a Mapping, got {type(identity).__name__}")
    resolved = dict(identity)
    if require_identity:
        if not resolved:
            raise V542AssemblyError(
                "V542_IDENTITY_ABSENT",
                "require_identity=True but the identity mapping is empty")
        issues = _identity_issues(resolved)
        if issues:
            raise V542AssemblyError(
                "V542_IDENTITY_DIRTY",
                "require_identity=True refused a dirty identity — "
                + "; ".join(sorted(issues)))
    return resolved


# ---------------------------------------------------------------------------
# the candidate-agnostic full model (WP1 forward REUSED by inheritance)
# ---------------------------------------------------------------------------
class AssembledV542Module(V542Model):
    """Full V5.4.2 model for ANY candidate — WP1 forward, per-candidate
    constructor.

    :class:`~pes2ts_core.pesn.models.v542.model.V542Model` freezes the
    forward surface, ``forward_move`` and the by-``id()`` param-group
    classifier; its ``__init__`` however asserts the A0-only migration
    invariants (``core == 55,392 + 49``), so a candidate-agnostic build
    cannot call it.  This subclass therefore initialises ``nn.Module``
    directly, registers the SAME three children in the SAME order
    (``backbone`` / ``move_decoder`` / ``process_head`` — the state-dict
    layout is byte-identical to ``V542Model``'s 45 keys) and re-expresses
    the constructor contract per candidate:

    * frozen decoder seam (:func:`assert_decoder_a_ref_frozen`, typed
      ``V542_DECODER_A_REF``) and the frozen head type
      (``V542_PROCESS_HEAD``);
    * the shared-``w_v`` invariant whenever the backbone aliases the
      decoder's ``w_v`` (``backbone.w_v is move_decoder.w_v``, typed
      ``V542_WV_NOT_SHARED`` — a forked ``w_v`` would double-update in
      the optimizer);
    * no ``torch.compile`` anywhere in the tree
      (``V542_TORCH_COMPILE_FORBIDDEN``);
    * the exact budgets for THIS candidate: ``head == 2,178``,
      ``core == expected_active + 49``,
      ``joint == expected_active + 49 + 2,178`` and
      ``joint <= 72,227`` (typed ``V542_PARAM_BUDGET``).
    """

    def __init__(self, backbone: nn.Module, *,
                 move_decoder: DirectMoveDecoder,
                 process_head: ReactionProcessHeadV53MP,
                 expected_active: int) -> None:
        # Deliberately NOT V542Model.__init__ (A0-only invariants): the
        # candidate-agnostic constructor is exactly this re-expression.
        nn.Module.__init__(self)
        if not isinstance(move_decoder, DirectMoveDecoder):
            raise V542InitError(
                "V542_DECODER_TYPE",
                "move_decoder must be the shared DirectMoveDecoder "
                f"(got {type(move_decoder).__name__})")
        assert_decoder_a_ref_frozen(move_decoder)
        if not isinstance(process_head, ReactionProcessHeadV53MP):
            raise V542InitError(
                "V542_PROCESS_HEAD",
                "process_head must be the frozen ReactionProcessHeadV53MP "
                f"(got {type(process_head).__name__})")
        if not isinstance(backbone, nn.Module):
            raise V542InitError(
                "V542_BACKBONE_TYPE",
                f"backbone must be an nn.Module, got {type(backbone).__name__}")
        if hasattr(backbone, "w_v") and \
                backbone.w_v is not move_decoder.w_v:
            raise V542InitError(
                "V542_WV_NOT_SHARED",
                "backbone.w_v is not the decoder's w_v module object — the "
                "shared-w_v invariant (backbone.w_v is move_decoder.w_v) must "
                "hold; pass the decoder that owns the backbone")
        self.backbone = backbone
        self.move_decoder = move_decoder
        self.process_head = process_head

        assert_no_torch_compile(self)
        self._assert_param_budget(int(expected_active))

    # -- candidate-agnostic budgets (the V542Model A0 invariants, re-expressed)
    def _assert_param_budget(self, expected_active: int) -> None:
        groups = self.param_groups_by_id()          # typed refusals, reused
        core = sum(int(p.numel()) for p in groups["core"])
        head = sum(int(p.numel()) for p in groups["head"])
        joint = self.n_parameters
        expected_core = expected_active + N_DECODER_PARAMS_V542
        expected_joint = expected_core + N_PROCESS_PARAMS_V542
        if head != N_PROCESS_PARAMS_V542:
            raise V542InitError(
                "V542_PARAM_BUDGET",
                f"head group has {head} params, expected "
                f"{N_PROCESS_PARAMS_V542}")
        if core != expected_core:
            raise V542InitError(
                "V542_PARAM_BUDGET",
                f"core group has {core} params, expected "
                f"{expected_active}+{N_DECODER_PARAMS_V542}={expected_core}")
        if joint != expected_joint:
            raise V542InitError(
                "V542_PARAM_BUDGET",
                f"joint params {joint} != {expected_active}+"
                f"{N_DECODER_PARAMS_V542}+{N_PROCESS_PARAMS_V542}"
                f"={expected_joint}")
        if joint > JOINT_PARAM_MAX_V542:
            raise V542InitError(
                "V542_PARAM_BUDGET",
                f"joint params {joint} > frozen joint_max "
                f"{JOINT_PARAM_MAX_V542}")

    # -- accounting ---------------------------------------------------------
    def parameter_report(self) -> Dict[str, int]:
        """Per-component counts WITHOUT the A0-only assertion of
        ``V542Model.parameter_report`` (assembly asserts per candidate)."""
        groups = self.param_groups_by_id()
        active = int(self.backbone.active_param_count)
        core = sum(int(p.numel()) for p in groups["core"])
        return {
            "backbone_active": active,
            "decoder": core - active,
            "process_head": sum(int(p.numel()) for p in groups["head"]),
            "joint": int(self.n_parameters),
        }


# ---------------------------------------------------------------------------
# the assembly record
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AssembledV542Model:
    """Auditable record returned by :func:`assemble_v542_model`.

    * ``model`` — the trainable :class:`AssembledV542Module` (frozen
      five-key forward; state-dict layout identical to ``V542Model``);
    * ``backbone`` / ``move_decoder`` / ``process_head`` — the SAME
      module objects the model registers (shared by reference, never
      copied);
    * ``candidate_id`` / ``run_seed`` — what was assembled;
    * ``built`` — the :class:`BuiltCandidate` audit record of the
      construction (actual ``MappingAudit``, construction seed, executed
      / deferred protocol steps);
    * ``seed_init`` — the construction seed of the frozen recipe
      ``int(sha256("v542-init|{run_seed}|{init_source_id}")…) % 2^31-1``;
    * ``anchor_seed`` — the deterministic process-head anchor seed (IS
      the run seed, frozen to ``{17, 29, 43}``);
    * ``param_groups`` — BY ``id()``: ``{"name": "core", "lr": 1e-4,
      "params": backbone ∪ move_decoder}`` first and ``{"name": "head",
      "lr": 3e-4, "params": process_head}`` second; every unique
      parameter exactly once, no duplicate ids (the shared ``w_v`` is one
      tensor object in exactly one group);
    * ``active_params`` / ``joint_params`` — the asserted exact counts;
    * ``identity`` / ``identity_sha256`` — the identity dict recorded
      VERBATIM (whatever the caller passed; ``{}`` when absent) and its
      canonical digest.
    """

    model: AssembledV542Module
    backbone: nn.Module
    move_decoder: DirectMoveDecoder
    process_head: ReactionProcessHeadV53MP
    candidate_id: str
    run_seed: int
    built: BuiltCandidate
    seed_init: int
    anchor_seed: int
    init_source_id: str
    param_groups: Tuple[Dict[str, Any], ...]
    active_params: int
    joint_params: int
    identity: Dict[str, Any]
    identity_sha256: str

    def group(self, name: str) -> list:
        """Parameters of one named group (``"core"`` / ``"head"``)."""
        for entry in self.param_groups:
            if entry["name"] == name:
                return list(entry["params"])
        raise V542AssemblyError(
            "V542_PARAM_GROUP_UNKNOWN",
            f"unknown param group {name!r} (known: "
            f"{[e['name'] for e in self.param_groups]})")

    def optimizer_param_groups(self) -> list:
        """FRESH optimizer-ready group dicts (never the record's own
        dicts — ``torch.optim`` MUTATES the dicts it is handed)."""
        return [{"name": str(entry["name"]), "lr": float(entry["lr"]),
                 "params": list(entry["params"])}
                for entry in self.param_groups]

    @property
    def core_params(self) -> list:
        return self.group("core")

    @property
    def head_params(self) -> list:
        return self.group("head")


# ---------------------------------------------------------------------------
# A0 baseline record (the registry ``build_candidate`` whitelist is the
# A1–A5 + A1-C/A5-C set; A0 follows the SAME frozen protocol locally)
# ---------------------------------------------------------------------------
def _build_a0_record(run_seed: int, move_decoder: DirectMoveDecoder, *,
                     r3_state: Optional[Mapping[str, torch.Tensor]] = None
                     ) -> BuiltCandidate:
    """A0 under the frozen construction protocol (steps 1→4, audited).

    Steps (inside :func:`construction_scope`, seeded
    ``seed_init(run_seed, "A0")``):

    1. fresh :class:`A0LegacyBackbone` (legacy ``StarEGNNV3`` construct
       order, ``w_v`` slot adopted from the SHARED decoder);
    2. the frozen A0 mapping — IDENTITY over every R3 backbone key
       except ``w_v.*``, whose two payload keys are declared in
       ``consumed_source_keys`` (they are consumed by step 4 from the
       SAME resolved payload — the strict audit counts every one of the
       37 source keys exactly once, no empty slots);
    3. no special zero-init for A0 (the migration baseline is R3-exact);
    4. :func:`apply_shared_w_v` — loads the SHA-verified R3 ``w_v`` into
       the shared decoder slot and asserts
       ``backbone.w_v is move_decoder.w_v``.

    Post-conditions (typed refusals): the backbone state dict is EXACTLY
    the frozen 37-key layout and every key is ``torch.equal`` to the R3
    payload; the exact A0 active count (55,392) is asserted.  Returns
    the frozen :class:`BuiltCandidate` record with steps 1–4 executed
    and steps 5–6 delegated (config ``construction_order``).
    """
    source = (load_r3_backbone_state() if r3_state is None
              else dict(r3_state))
    with construction_scope(run_seed, A0) as init_seed:
        backbone = A0LegacyBackbone(move_decoder)
        mapping = {key: key for key in source
                   if key not in _A0_WV_SOURCE_KEYS}
        audit = apply_r3_mapping(
            backbone, source, mapping,
            consumed_source_keys=list(_A0_WV_SOURCE_KEYS))
        apply_shared_w_v(backbone, move_decoder, source=source)

    state = backbone.state_dict()
    if set(state) != set(source) or len(state) != _A0_BACKBONE_KEYS:
        raise V542InitError(
            "V542_A0_R3_LAYOUT",
            f"A0 backbone state has {len(state)} keys, the frozen R3 "
            f"layout is {_A0_BACKBONE_KEYS} identity keys — got "
            f"missing={sorted(set(source) - set(state))} "
            f"extra={sorted(set(state) - set(source))}")
    diverged = sorted(key for key in state
                      if not torch.equal(state[key], source[key]))
    if diverged:
        raise V542InitError(
            "V542_A0_R3_NOT_EQUAL",
            "A0 migration invariant failed: keys not torch.equal to the "
            f"R3 payload: {diverged}")
    assert_candidate_budget(backbone, expected_active=A0_ACTIVE_PARAMS_V542)
    return BuiltCandidate(
        candidate_id=A0,
        backbone=backbone,
        move_decoder=move_decoder,
        mapping_audit=audit,
        seed_init=init_seed,
        init_source_id=A0,
        executed_steps=CONSTRUCTION_ORDER[:4],
    )


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------
def assemble_v542_model(
        candidate_id: str,
        run_seed: int,
        *,
        device: Any,
        identity: Optional[Mapping[str, Any]] = None,
        require_identity: bool = False,
        r3_state: Optional[Mapping[str, torch.Tensor]] = None,
) -> AssembledV542Model:
    """Assemble the FULL model of ``candidate_id`` for ``run_seed``.

    Frozen assembly order (config ``init_policy.construction_order``)::

        (in ONE torch.random.fork_rng(devices=[]) scope — the caller's
         RNG stream is restored byte-identically on exit AND on error)
          fresh DirectMoveDecoder                       [P3 seam]
          build_candidate / _build_a0_record            [steps 1–4]
          build_process_head(anchor_seed = run_seed)    [step 5]
        finish_training_seed(run_seed)                  [step 6]
        verify torch.initial_seed() == run_seed

    Args:
        candidate_id: ``A0``, ``A1``–``A5`` or ``A1-C``/``A5-C``; an
            unknown id is a typed ``V542_CANDIDATE_UNKNOWN`` refusal (a
            runner must never guess a family).
        run_seed: the run seed AND the process-head anchor seed — only
            the frozen ``{17, 29, 43}`` are accepted, anything else is a
            typed ``V542_ANCHOR_SEED`` refusal BEFORE any build.
        device: target device (``torch.device`` or ``"cpu"`` /
            ``"cuda…"``); a CUDA device without CUDA is typed-refused.
        identity: run-identity SHA block to RECORD verbatim (also
            digest).  Absent/dirty values are refused only when
            ``require_identity=True`` (typed ``V542_IDENTITY_ABSENT`` /
            ``V542_IDENTITY_DIRTY``); otherwise whatever dict was
            passed is recorded as given (``{}`` when ``None``).
        require_identity: opt into the strict identity check (the runner
            lane's mode).
        r3_state: inject an already-resolved 37-key R3 payload (tests);
            ``None`` re-verifies the pinned R3 file bytes on every call.

    Returns:
        The :class:`AssembledV542Model` record with the exact asserted
        counts and the by-``id()`` param groups.
    """
    # -- 0. anchor-seed rule (typed refusal BEFORE any construction) -------
    if not isinstance(run_seed, int) or isinstance(run_seed, bool) or \
            int(run_seed) not in ANCHOR_SEEDS_V542:
        raise V542InitSeedError(
            "V542_ANCHOR_SEED",
            f"run_seed/anchor_seed must be one of the frozen seeds "
            f"{ANCHOR_SEEDS_V542} (config process_head_anchor.seeds — the "
            f"anchor seed IS the run seed), got {run_seed!r}")
    run_seed = int(run_seed)

    # -- 0b. candidate id + its frozen EXACT active count ------------------
    if not isinstance(candidate_id, str) or candidate_id not in \
            V542_CANDIDATE_IDS_V542:
        raise V542InitMapError(
            "V542_CANDIDATE_UNKNOWN",
            f"unknown candidate id {candidate_id!r}; known: "
            f"{sorted(V542_CANDIDATE_IDS_V542)}")
    expected_active = (A0_ACTIVE_PARAMS_V542 if candidate_id == A0
                       else CANDIDATE_PARAM_COUNTS[candidate_id])

    # -- 0c. identity (recorded verbatim; strict only on demand) -----------
    recorded_identity = _resolve_identity(
        identity, require_identity=bool(require_identity))

    # -- 0d. device ---------------------------------------------------------
    device_str = str(device)
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise V542AssemblyError(
            "V542_CUDA_UNAVAILABLE",
            f"device={device_str!r} but CUDA is not available")

    # -- 1–5. construction, RNG-isolated ----------------------------------
    with torch.random.fork_rng(devices=[]):
        move_decoder = DirectMoveDecoder()          # fresh (always R3-loaded)
        if candidate_id == A0:
            built = _build_a0_record(run_seed, move_decoder,
                                     r3_state=r3_state)
        else:
            built = build_candidate(candidate_id, run_seed, move_decoder,
                                    r3_state=r3_state)
        process_head = build_process_head(run_seed)  # step 5, anchor = run seed
        model = AssembledV542Module(
            built.backbone, move_decoder=move_decoder,
            process_head=process_head, expected_active=expected_active)

    # -- 6. training-seed reset, VERIFIED (never assumed) -------------------
    finish_training_seed(run_seed)
    if int(torch.initial_seed()) != run_seed:
        raise V542InitSeedError(  # pragma: no cover - finish_* already checks
            "V542_INIT_SEED_RESET",
            f"torch.initial_seed() is {torch.initial_seed()} after "
            f"finish_training_seed({run_seed}) — the training seed was "
            "not reset")

    # -- exact counts, once more against the frozen registry ---------------
    active = assert_candidate_budget(built.backbone,
                                     expected_active=expected_active)
    groups = model.param_groups_by_id()
    core_n = sum(int(p.numel()) for p in groups["core"])
    head_n = sum(int(p.numel()) for p in groups["head"])
    joint = model.n_parameters
    if head_n != N_PROCESS_PARAMS_V542 or \
            core_n != active + N_DECODER_PARAMS_V542:
        raise V542AssemblyError(  # pragma: no cover - module asserts too
            "V542_PARAM_BUDGET",
            f"param groups drifted: core={core_n} head={head_n} for "
            f"active={active}")
    if joint > JOINT_PARAM_MAX_V542:
        raise V542AssemblyError(
            "V542_PARAM_BUDGET",
            f"{candidate_id} joint params {joint} > frozen joint_max "
            f"{JOINT_PARAM_MAX_V542}")
    if candidate_id == A0 and joint != A0_JOINT_PARAMS_V542:
        raise V542AssemblyError(  # pragma: no cover - migration invariant
            "V542_PARAM_BUDGET",
            f"A0 joint {joint} != migration invariant "
            f"{A0_JOINT_PARAMS_V542}")

    # -- by-id param groups with the frozen lrs ----------------------------
    param_groups: Tuple[Dict[str, Any], ...] = (
        {"name": "core", "lr": LR_CORE_V542,
         "params": list(groups["core"])},
        {"name": "head", "lr": LR_HEAD_V542,
         "params": list(groups["head"])},
    )
    seen = [id(p) for entry in param_groups for p in entry["params"]]
    model_ids = {id(p) for p in model.parameters()}
    if len(seen) != len(set(seen)) or set(seen) != model_ids:
        raise V542AssemblyError(  # pragma: no cover - param_groups_by_id
            "V542_PARAM_GROUP_DUPLICATE",       # enforces the same partition
            "param groups must hold every unique model parameter exactly "
            "once (duplicate or unclassified ids found)")

    model.to(device)

    return AssembledV542Model(
        model=model,
        backbone=built.backbone,
        move_decoder=built.move_decoder,
        process_head=model.process_head,
        candidate_id=candidate_id,
        run_seed=run_seed,
        built=built,
        seed_init=int(built.seed_init),
        anchor_seed=run_seed,
        init_source_id=str(built.init_source_id),
        param_groups=param_groups,
        active_params=int(active),
        joint_params=int(joint),
        identity=recorded_identity,
        identity_sha256=identity_sha256(recorded_identity),
    )
