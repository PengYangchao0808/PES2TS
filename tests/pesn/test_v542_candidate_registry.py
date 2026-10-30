"""Frozen candidate registry contract (V5.4.2 WP2; gate-2 remediation).

Locks the single entry point a runner may use: the seven frozen candidate
ids (A1, A1-C, A2, A3, A4, A5, A5-C), their exact measured active
parameter counts, the winner-eligible vs diagnostic split, and the typed
refusal for an unknown id.  Construction here goes through the frozen
``init_policy.construction_order`` protocol (fork_rng + seed recipe + R3
mapping + zero-inits) and returns the audit-bearing ``BuiltCandidate``.

Gate-2 (oracle-gate2.md) guarantees locked here:

* **B1** — EVERY builder loads the SHA-verified R3 ``w_v`` into the
  shared decoder: a freshly constructed ``DirectMoveDecoder()`` ends up
  ``torch.equal`` to R3 after ``build_candidate`` for all seven ids, and
  the result is ORDER-INDEPENDENT (A2 built first in a fresh order, then
  A1, both R3-equal);
* **B2** — a tampered frozen ``a_ref`` (and a non-decoder object) is a
  typed refusal for all seven ids at the ``build_candidate`` seam;
* **C1** — an injected synthetic ``r3_state`` feeds BOTH the backbone
  mapping and the decoder ``w_v`` (decoder ``w_v == source["w_v.*"]``);
* **C3** — ``build_candidate`` returns the frozen ``BuiltCandidate``
  record: non-empty ``MappingAudit`` from the actual mapping call, a
  ``seed_init`` that matches the frozen recipe, and an audit digest that
  is stable across two identical builds;
* **C4** — explicit MOVE ownership: ``move_decoder.w_v`` equals R3 after
  build; ``backbone.w_v`` is optional but, if present, must be the SAME
  module object.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.models.key_atom_egnn_v3 import (  # noqa: E402
    StarEGNNV3,
)
from pes2ts_core.pesn.models.v542 import V542InitError  # noqa: E402
from pes2ts_core.pesn.models.v542.candidates import (  # noqa: E402
    BUILDER_DELEGATED_STEPS,
    CANDIDATE_BUILDERS,
    CANDIDATE_IDS,
    CANDIDATE_PARAM_COUNTS,
    CONSTRUCTION_ORDER,
    DIAGNOSTIC_CANDIDATE_IDS,
    BuiltCandidate,
    V542InitMapError,
    build_candidate,
    finish_training_seed,
    load_r3_backbone_state,
    resolve_init_source_id,
    seed_init,
)
from pes2ts_core.pesn.models.v542.decoder import DirectMoveDecoder  # noqa: E402

SEED = 17

#: The exact canonical-protocol steps each candidate builder EXECUTES
#: (config ``init_policy.construction_order`` steps 1–4: fresh modules,
#: R3 mapping, per-candidate zero-init, shared ``w_v`` from R3).  The
#: zero-init variant differs per candidate but is ONE canonical step;
#: no budget step, no builder-specific drift — steps 5–6 are always
#: builder-delegated.
EXPECTED_EXECUTED_STEPS = {
    candidate_id: CONSTRUCTION_ORDER[:4] for candidate_id in CANDIDATE_BUILDERS
}
#: The exact canonical steps every builder DEFERS (process head is
#: assembled by P3; ``finish_training_seed`` is the caller's job).
EXPECTED_DEFERRED_STEPS = (
    "5_process_head_from_deterministic_anchor_replay",
    "6_set_seed(run_seed)_before_training",
)


@pytest.fixture(scope="module")
def r3_state():
    return load_r3_backbone_state()


def test_registry_covers_exactly_the_frozen_ids():
    assert set(CANDIDATE_BUILDERS) == {
        "A1", "A1-C", "A2", "A3", "A4", "A5", "A5-C",
    }
    assert set(CANDIDATE_IDS) == {"A1", "A2", "A3", "A4", "A5"}
    assert set(DIAGNOSTIC_CANDIDATE_IDS) == {"A1-C", "A5-C"}
    # diagnostics never appear in the winner-eligible tuple
    assert not set(CANDIDATE_IDS) & set(DIAGNOSTIC_CANDIDATE_IDS)


def test_registry_param_counts_are_the_measured_values():
    assert CANDIDATE_PARAM_COUNTS == {
        "A1": 41_232,
        "A1-C": 57_264,
        "A2": 54_896,
        "A3": 53_328,
        "A4": 55_840,
        "A5": 31_584,
        "A5-C": 53_712,
    }


# ---------------------------------------------------------------------------
# C4: frozen count + EXPLICIT move-ownership assertion (no hasattr tolerance)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("candidate_id", sorted(CANDIDATE_BUILDERS))
def test_build_candidate_hits_the_frozen_count(candidate_id, r3_state):
    decoder = DirectMoveDecoder()
    record = build_candidate(candidate_id, SEED, decoder)
    finish_training_seed(SEED)
    assert isinstance(record, BuiltCandidate)
    assert record.candidate_id == candidate_id
    backbone = record.backbone
    measured = (
        backbone.active_param_count
        if hasattr(backbone, "active_param_count")
        else sum(p.numel() for p in backbone.parameters())
    )
    assert int(measured) == CANDIDATE_PARAM_COUNTS[candidate_id]
    # MOVE ownership: the decoder always owns the 49-param w_v and it is
    # the R3 load (gate-2 B1/C4) — checked by VALUE, not by presence.
    assert int(decoder.param_count) == 49
    assert torch.equal(decoder.w_v.weight, r3_state["w_v.weight"])
    assert torch.equal(decoder.w_v.bias, r3_state["w_v.bias"])
    # the optional backbone alias must be the SAME module object
    if hasattr(backbone, "w_v"):
        assert backbone.w_v is decoder.w_v
    assert record.move_decoder is decoder


# ---------------------------------------------------------------------------
# B1: a FRESH decoder is R3-loaded by every builder (all seven ids)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("candidate_id", sorted(CANDIDATE_BUILDERS))
def test_build_candidate_r3_loads_fresh_decoder_w_v(candidate_id, r3_state):
    decoder = DirectMoveDecoder()
    random_before = decoder.w_v.weight.detach().clone()
    build_candidate(candidate_id, SEED, decoder)
    # the build genuinely LOADED the R3 w_v (old A2/A5/A5-C left the
    # fresh random draw untouched — the order-dependence blocker)
    assert not torch.equal(decoder.w_v.weight, random_before)
    assert torch.equal(decoder.w_v.weight, r3_state["w_v.weight"])
    assert torch.equal(decoder.w_v.bias, r3_state["w_v.bias"])


def test_build_order_independent_a2_first_then_a1(r3_state):
    """Fresh order: A2 (the former blocker) built BEFORE A1 — both
    decoders must end R3-equal, so no candidate's w_v depends on build
    order (gate-2 B1)."""
    decoder_a2 = DirectMoveDecoder()
    record_a2 = build_candidate("A2", SEED, decoder_a2)
    decoder_a1 = DirectMoveDecoder()
    record_a1 = build_candidate("A1", SEED, decoder_a1)
    assert record_a2.candidate_id == "A2"
    assert record_a1.candidate_id == "A1"
    for cid, decoder in (("A2", decoder_a2), ("A1", decoder_a1)):
        assert torch.equal(decoder.w_v.weight, r3_state["w_v.weight"]), cid
        assert torch.equal(decoder.w_v.bias, r3_state["w_v.bias"]), cid
    # identical shared w_v across candidates regardless of order
    assert torch.equal(decoder_a2.w_v.weight, decoder_a1.w_v.weight)
    assert torch.equal(decoder_a2.w_v.bias, decoder_a1.w_v.bias)
    # and the same holds for a decoder REUSED across candidates
    build_candidate("A5", SEED, decoder_a2)
    assert torch.equal(decoder_a2.w_v.weight, r3_state["w_v.weight"])
    assert torch.equal(decoder_a2.w_v.bias, r3_state["w_v.bias"])


# ---------------------------------------------------------------------------
# B2: the frozen a_ref seam is enforced at build_candidate (all seven)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("candidate_id", sorted(CANDIDATE_BUILDERS))
def test_tampered_a_ref_is_refused_for_every_candidate(candidate_id):
    decoder = DirectMoveDecoder()
    decoder._a_ref = 0.1                       # tamper attempt
    with pytest.raises(V542InitError) as exc:
        build_candidate(candidate_id, SEED, decoder)
    assert exc.value.code == "V542_DECODER_A_REF"


@pytest.mark.parametrize("candidate_id", sorted(CANDIDATE_BUILDERS))
def test_non_decoder_is_refused_for_every_candidate(candidate_id):
    with pytest.raises(V542InitError) as exc:
        build_candidate(candidate_id, SEED, object())  # type: ignore[arg-type]
    assert exc.value.code == "V542_DECODER_A_REF"


# ---------------------------------------------------------------------------
# C1: an injected synthetic r3_state feeds the backbone AND the decoder
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "candidate_id", ["A1", "A1-C", "A2", "A3", "A4", "A5", "A5-C"])
def test_injected_synthetic_state_feeds_decoder_w_v(candidate_id):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4242)
        synthetic = StarEGNNV3().state_dict()   # 37-key R3-like payload
    assert "w_v.weight" in synthetic and "w_v.bias" in synthetic
    decoder = DirectMoveDecoder()
    record = build_candidate(candidate_id, SEED, decoder,
                             r3_state=synthetic)
    # the SAME resolved source feeds the backbone mapping …
    mapped = record.mapping_audit.source_keys
    assert set(mapped) == set(synthetic)
    # … and the decoder w_v (C1: never the pinned file for one and the
    # injected payload for the other)
    assert torch.equal(decoder.w_v.weight, synthetic["w_v.weight"])
    assert torch.equal(decoder.w_v.bias, synthetic["w_v.bias"])
    if hasattr(record.backbone, "w_v"):
        assert torch.equal(record.backbone.w_v.weight,
                           synthetic["w_v.weight"])


# ---------------------------------------------------------------------------
# C3: build_candidate returns the audit-bearing BuiltCandidate record
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("candidate_id", sorted(CANDIDATE_BUILDERS))
def test_build_candidate_returns_non_empty_audit_record(candidate_id):
    decoder = DirectMoveDecoder()
    record = build_candidate(candidate_id, SEED, decoder)
    assert isinstance(record, BuiltCandidate)
    assert record.candidate_id == candidate_id
    assert record.move_decoder is decoder
    audit = record.mapping_audit
    # the audit came from the ACTUAL mapping call: the full 37-key
    # source was audited, at least one target was written and verified
    assert len(audit.source_keys) == 37
    assert len(audit.target_keys) >= 1
    assert set(audit.torch_equal_verified) == set(audit.target_keys)
    assert audit.missing == () and audit.unexpected == ()
    assert audit.shape_mismatch == () and audit.dtype_mismatch == ()
    assert audit.mapping_table_sha256.startswith("LOCAL_PLACEHOLDER:")
    # seed identity: exactly the frozen recipe
    assert record.seed_init == seed_init(
        SEED, resolve_init_source_id(candidate_id))
    assert record.init_source_id == resolve_init_source_id(candidate_id)
    # the executed step list is recorded and non-empty
    assert isinstance(record.construction_order, tuple)
    assert len(record.construction_order) >= 4
    assert all(isinstance(step, str) and step
               for step in record.construction_order)


# ---------------------------------------------------------------------------
# C3 residual: construction_order IS the canonical frozen six-step protocol
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("candidate_id", sorted(CANDIDATE_BUILDERS))
def test_construction_order_is_the_canonical_frozen_six(candidate_id):
    """Every builder records the CANONICAL ``init_policy.
    construction_order`` 6-tuple verbatim — never a builder-specific
    list — with per-step status in ``executed_steps`` / ``deferred_steps``
    (gate-2 C3 residual condition)."""
    record = build_candidate(candidate_id, SEED, DirectMoveDecoder())
    # the canonical tuple, EXACTLY — same length (6) and same names for
    # every candidate; no budget step, no omissions, no reordering
    assert record.construction_order == CONSTRUCTION_ORDER
    assert len(record.construction_order) == len(CONSTRUCTION_ORDER) == 6
    # executed: exactly steps 1–4 (fresh, mapping, zero-init, w_v) —
    # the per-builder subset, no budget gate in the list
    assert record.executed_steps == EXPECTED_EXECUTED_STEPS[candidate_id]
    assert all(isinstance(step, str) and step
               for step in record.executed_steps)
    assert "assert_candidate_budget" not in record.executed_steps
    # deferred: exactly the two builder-delegated canonical steps
    # (process head by P3, training-seed reset by the caller)
    assert record.deferred_steps == EXPECTED_DEFERRED_STEPS
    assert record.deferred_steps == BUILDER_DELEGATED_STEPS
    assert set(record.deferred_steps) == set(CONSTRUCTION_ORDER[4:6])
    # executed + deferred partition the canonical protocol
    assert set(record.executed_steps) | set(record.deferred_steps) \
        == set(CONSTRUCTION_ORDER)
    assert not set(record.executed_steps) & set(record.deferred_steps)


@pytest.mark.parametrize("candidate_id", sorted(CANDIDATE_BUILDERS))
def test_audit_digest_stable_across_identical_builds(candidate_id):
    rec_a = build_candidate(candidate_id, SEED, DirectMoveDecoder())
    rec_b = build_candidate(candidate_id, SEED, DirectMoveDecoder())
    assert rec_a.seed_init == rec_b.seed_init
    assert rec_a.construction_order == rec_b.construction_order
    assert rec_a.executed_steps == rec_b.executed_steps
    assert rec_a.deferred_steps == rec_b.deferred_steps
    assert rec_a.mapping_audit == rec_b.mapping_audit
    assert (rec_a.mapping_audit.mapping_table_sha256
            == rec_b.mapping_audit.mapping_table_sha256)


def test_unknown_candidate_is_a_typed_refusal():
    with pytest.raises(V542InitMapError) as excinfo:
        build_candidate("A9", SEED, DirectMoveDecoder())
    assert excinfo.value.code == "V542_CANDIDATE_UNKNOWN"
