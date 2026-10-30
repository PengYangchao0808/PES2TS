"""V5.4.2 shared candidate scaffolding (WP2 lane P2-0; ADR-0033 §2.3).

Every A1–A5 lane imports from THIS package — the frozen surface that
makes the candidates comparable under one pre-registered construction
protocol (config ``training_contract.init_policy``)::

    from pes2ts_core.pesn.models.v542.candidates import (
        R3_CHECKPOINT_PATH, R3_CHECKPOINT_SHA256,
        INIT_SEED_TEMPLATE, INIT_SOURCE_ID_OVERRIDES,
        seed_init, resolve_init_source_id, load_r3_backbone_state,
        construction_scope, finish_training_seed,
        apply_r3_mapping, MappingAudit, zero_init_,
        apply_r3_w_v, apply_shared_w_v, build_process_head,
        BuiltCandidate, build_candidate,
        assert_candidate_budget, check_backbone_contract,
        V542InitMapError, V542InitSeedError, V542CandidateBudgetError,
    )

Scope discipline: this package only ADDS shared scaffolding — no
candidate architecture lives here, no existing V54.2 file
(``interfaces``/``decoder``/``backbone_a0``/``loader``/``model``) is
modified, and nothing here calls torch.compile (ADR-0033 §2.1).  The
implementation and its audit semantics are documented in
:mod:`pes2ts_core.pesn.models.v542.candidates.common`.
"""

from pes2ts_core.pesn.models.v542.candidates.common import (
    BUILDER_DELEGATED_STEPS,
    CONSTRUCTION_ORDER,
    INIT_SEED_TEMPLATE,
    INIT_SOURCE_ID_OVERRIDES,
    R3_CHECKPOINT_PATH,
    R3_CHECKPOINT_SHA256,
    STEP_FRESH_MODULES,
    STEP_PROCESS_HEAD,
    STEP_R3_MAPPING,
    STEP_SHARED_W_V,
    STEP_TRAINING_SEED,
    STEP_ZERO_INIT,
    BuiltCandidate,
    MappingAudit,
    V542CandidateBudgetError,
    V542InitMapError,
    V542InitSeedError,
    apply_r3_mapping,
    apply_r3_w_v,
    apply_shared_w_v,
    assert_candidate_budget,
    build_process_head,
    check_backbone_contract,
    construction_scope,
    finish_training_seed,
    load_r3_backbone_state,
    resolve_init_source_id,
    seed_init,
    zero_init_,
)
from pes2ts_core.pesn.models.v542.decoder import (  # gate-1 B1 / gate-2 B2
    assert_decoder_a_ref_frozen,
)

__all__ = [
    "BUILDER_DELEGATED_STEPS",
    "CANDIDATE_BUILDERS",
    "CANDIDATE_IDS",
    "CANDIDATE_PARAM_COUNTS",
    "CONSTRUCTION_ORDER",
    "DIAGNOSTIC_CANDIDATE_IDS",
    "INIT_SEED_TEMPLATE",
    "INIT_SOURCE_ID_OVERRIDES",
    "R3_CHECKPOINT_PATH",
    "R3_CHECKPOINT_SHA256",
    "BuiltCandidate",
    "MappingAudit",
    "V542CandidateBudgetError",
    "V542InitMapError",
    "V542InitSeedError",
    "apply_r3_mapping",
    "apply_r3_w_v",
    "apply_shared_w_v",
    "assert_candidate_budget",
    "build_candidate",
    "build_process_head",
    "check_backbone_contract",
    "construction_scope",
    "finish_training_seed",
    "load_r3_backbone_state",
    "resolve_init_source_id",
    "seed_init",
    "zero_init_",
]


# ---------------------------------------------------------------------------
# Candidate registry (WP2): the single frozen entry point a runner may use.
# Each factory returns the audit-bearing :class:`BuiltCandidate` record.
# ---------------------------------------------------------------------------
def _build_a1(run_seed, move_decoder, *, r3_state=None):
    return _a1.build_record(run_seed, move_decoder, r3_state=r3_state,
                            layers=2)


def _build_a1c(run_seed, move_decoder, *, r3_state=None):
    return _a1.build_record(run_seed, move_decoder, r3_state=r3_state,
                            layers=3)


def _build_a2(run_seed, move_decoder, *, r3_state=None):
    return _a2.build_record(run_seed, move_decoder, r3_state=r3_state)


def _build_a3(run_seed, move_decoder, *, r3_state=None):
    return _a3.build_record(run_seed, move_decoder, r3_state=r3_state)


def _build_a4(run_seed, move_decoder, *, r3_state=None):
    return _a4.build_record(run_seed, move_decoder, r3_state=r3_state)


def _build_a5(run_seed, move_decoder, *, r3_state=None):
    return _a5.build_record(run_seed, move_decoder, r3_state=r3_state,
                            cells=1)


def _build_a5c(run_seed, move_decoder, *, r3_state=None):
    return _a5.build_record(run_seed, move_decoder, r3_state=r3_state,
                            cells=2)


#: Frozen candidate id -> factory (``(run_seed, move_decoder, *, r3_state)`).
CANDIDATE_BUILDERS = {
    "A1": _build_a1,
    "A1-C": _build_a1c,
    "A2": _build_a2,
    "A3": _build_a3,
    "A4": _build_a4,
    "A5": _build_a5,
    "A5-C": _build_a5c,
}

#: Winner-eligible main-matrix candidates (diagnostics excluded).
CANDIDATE_IDS: tuple[str, ...] = ("A1", "A2", "A3", "A4", "A5")

#: Diagnostic-only attribution candidates (never winner-eligible).
DIAGNOSTIC_CANDIDATE_IDS: tuple[str, ...] = ("A1-C", "A5-C")

#: Exact measured active-backbone parameter counts (asserted by the lanes;
#: equal to the preregistered ``backbone_active`` estimates).
CANDIDATE_PARAM_COUNTS: dict[str, int] = {
    "A1": 41_232,
    "A1-C": 57_264,
    "A2": 54_896,
    "A3": 53_328,
    "A4": 55_840,
    "A5": 31_584,
    "A5-C": 53_712,
}


def build_candidate(
        candidate_id: str,
        run_seed: int,
        move_decoder,
        *,
        r3_state=None,
) -> BuiltCandidate:
    """Build ``candidate_id`` through the frozen construction protocol;
    returns the audit-bearing :class:`BuiltCandidate` record (backbone,
    shared decoder, the ACTUAL :class:`MappingAudit`, the construction
    seed, the resolved init source and the executed step list).

    BEFORE dispatch the gate-1 B1 / gate-2 B2 seam is frozen:
    ``move_decoder`` must carry the frozen ``a_ref=0.08``
    (:func:`~pes2ts_core.pesn.models.v542.decoder.
    assert_decoder_a_ref_frozen`, typed ``V542_DECODER_A_REF``) — a
    tampered decoder is refused for EVERY candidate id.  Unknown ids
    are a typed refusal — a runner must never guess a family.  The
    caller is responsible for ``finish_training_seed(run_seed)`` after
    construction (config ``init_policy.construction_order`` step 6).
    """
    assert_decoder_a_ref_frozen(move_decoder)
    try:
        factory = CANDIDATE_BUILDERS[candidate_id]
    except KeyError:
        raise V542InitMapError(
            "V542_CANDIDATE_UNKNOWN",
            f"unknown candidate id {candidate_id!r}; known: "
            f"{sorted(CANDIDATE_BUILDERS)}",
        ) from None
    return factory(run_seed, move_decoder, r3_state=r3_state)


from pes2ts_core.pesn.models.v542.candidates import (  # noqa: E402
    a1 as _a1,
    a2 as _a2,
    a3 as _a3,
    a4 as _a4,
    a5 as _a5,
)
