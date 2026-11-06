"""V5.4.3 WP6 — the PROCESS-track candidate package (plan §6.3 :538–556,
§11.1 :1104–1110): ``P1`` Residual MLP, ``P2`` DeepSets, ``P3`` Latent
Graph-GNN, ``P4`` Graph Transformer and its parameter-matched control ``P4-C``.

What this package owns
----------------------
* one file per candidate exactly as §11.1 sketches
  (``residual_mlp.py`` / ``deepsets.py`` / ``graph_gnn.py`` /
  ``graph_transformer.py``; ``graph_gru.py`` is ``P5``, a CONDITIONAL
  candidate that plan §6.3 :585 only activates after P1–P4 have run, so it is
  deliberately absent — declaring a stub would be exactly the "design
  placeholder" WP4's registry refuses);
* ``tokens.py``, the ONE shared token definition §6.3 :552 requires
  ("P2/P3/P4 必须使用同一 token 定义"); without it the three local candidates
  could differ in their input and H-P1/H-P2 (plan §3.2 :166–167) would be
  unattributable;
* the registration surface: :func:`process_candidate_specs` and
  :func:`register_process_candidates`.  Registration is EXPLICIT and
  side-effect free on import: the process-wide
  :data:`~pes2ts_core.pesn.models.v543.registry.REGISTRY_V543` is WP4's object
  and its "only the M0/P0 baselines" state is asserted by
  ``tests/pesn/test_v543_registry.py``, so this module never mutates it.  A
  caller (trainer, campaign script, test) passes the registry it owns.

Why ``P4-C`` is not a :class:`DecoderSpec`
------------------------------------------
Plan §6.3 :547 declares ``P4-C``, but ``registry.DECODER_IDS_V543`` declares
only ``P0…P5`` (:539–548) and both the spec and the protocol checker refuse an
undeclared id.  ``P4-C`` therefore carries its own
:class:`~pes2ts_core.pesn.models.v543.process.graph_transformer.P4ControlSpec`
with the same 13 §6.1 fields — a reported plan ↔ WP4 conflict, not a silent
workaround (see ``graph_transformer.py``'s module docstring).

Discipline: no candidate reads a batch/``pos``/supervision key (plan §6.1
:519), none calls ``torch.compile`` (plan §11.3), and this package never edits
a shared V543 file."""

from __future__ import annotations

from typing import Any, Final

import torch

from pes2ts_core.pesn.models.v543.process.deepsets import (
    DECODER_ID_P2,
    P2_EXPECTED_PARAMS,
    P2_SPEC,
    ProcessDeepSets,
    build_p2,
)
from pes2ts_core.pesn.models.v543.process.graph_gnn import (
    DECODER_ID_P3,
    P3_EXPECTED_PARAMS,
    P3_SPEC,
    ProcessLatentGraphGNN,
    build_p3,
)
from pes2ts_core.pesn.models.v543.process.graph_transformer import (
    DECODER_ID_P4,
    DECODER_ID_P4C,
    P4_EXPECTED_PARAMS,
    P4_SPEC,
    P4C_BAND_PCT,
    P4C_EXPECTED_PARAMS,
    P4C_SPEC,
    P4ControlSpec,
    ProcessGraphTransformer,
    ProcessP4Control,
    assert_p4c_matches_p4,
    build_p4,
    build_p4c,
    p4_reference_params,
    p4c_band,
)
from pes2ts_core.pesn.models.v543.process.residual_mlp import (
    DECODER_ID_P1,
    P1_EXPECTED_PARAMS,
    P1_SPEC,
    ProcessResidualMLP,
    build_p1,
)
from pes2ts_core.pesn.models.v543.process.tokens import (
    PROCESS_INIT_SEED_V543,
    PROCESS_TOKEN_DIM_V543,
    TokenEdges,
    TokenTensors,
    V543ProcessDecoderError,
    deterministic_init,
    local_token_edges,
    token_tensors,
)
from pes2ts_core.pesn.models.v543.registry import (
    TRACK_PROCESS,
    DecoderRegistry,
    DecoderSpec,
)

#: The four FORMAL PROCESS candidates of plan §6.3 :543–546 (P4-C excluded by
#: design: it is a DIAGNOSTIC control, :547 "诊断（仅 P4 进 top2 时）").
FORMAL_PROCESS_DECODER_IDS_V543: Final[tuple[str, ...]] = (
    DECODER_ID_P1, DECODER_ID_P2, DECODER_ID_P3, DECODER_ID_P4)
#: Every PROCESS candidate this package implements (formal + the control).
PROCESS_CANDIDATE_IDS_V543: Final[tuple[str, ...]] = (
    FORMAL_PROCESS_DECODER_IDS_V543 + (DECODER_ID_P4C,))

#: Candidate class per id (the control is reachable but not registrable).
PROCESS_CANDIDATE_CLASSES_V543: Final[dict[str, type]] = {
    DECODER_ID_P1: ProcessResidualMLP,
    DECODER_ID_P2: ProcessDeepSets,
    DECODER_ID_P3: ProcessLatentGraphGNN,
    DECODER_ID_P4: ProcessGraphTransformer,
    DECODER_ID_P4C: ProcessP4Control,
}

#: The plan §6.3 "目标参数" estimates (the REPORTED comparison column, never an
#: implementation-time assertion — plan §6.4 :569 "估算值不得作为实现前 exact
#: 断言").
PLAN_PARAM_ESTIMATES_V543: Final[dict[str, int]] = {
    DECODER_ID_P1: 36_500,
    DECODER_ID_P2: 35_700,
    DECODER_ID_P3: 34_500,
    DECODER_ID_P4: 37_000,
}
#: The §6.4 :569 exact counts of the implemented structures.
MEASURED_PARAMS_V543: Final[dict[str, int]] = {
    DECODER_ID_P1: P1_EXPECTED_PARAMS,
    DECODER_ID_P2: P2_EXPECTED_PARAMS,
    DECODER_ID_P3: P3_EXPECTED_PARAMS,
    DECODER_ID_P4: P4_EXPECTED_PARAMS,
    DECODER_ID_P4C: P4C_EXPECTED_PARAMS,
}


def process_candidate_specs() -> tuple[DecoderSpec, ...]:
    """The four FORMAL candidates' §6.1 registry specs (plan §6.3 :543–546)."""
    return (P1_SPEC, P2_SPEC, P3_SPEC, P4_SPEC)


def register_process_candidates(registry: DecoderRegistry
                                ) -> DecoderRegistry:
    """Register P1–P4 into ``registry`` (P0 must already be there).

    Registration is what performs the §6.4 construction assertion (WP4
    ``registry.register`` builds the decoder once and refuses an out-of-band
    count), so this is the canonical "register the PROCESS candidates" call for
    a campaign script or a test.  The process-wide ``REGISTRY_V543`` is NOT
    touched (see the module docstring).
    """
    if not isinstance(registry, DecoderRegistry):
        raise V543ProcessDecoderError(
            "V543_PROCESS_REGISTRY",
            f"registry must be a DecoderRegistry, got "
            f"{type(registry).__name__}")
    for spec in process_candidate_specs():
        registry.register(spec)
    return registry


def build_process_candidate(decoder_id: str, **kwargs: Any
                            ) -> torch.nn.Module:
    """Construct a PROCESS candidate by id (P4-C included).

    A thin convenience over :data:`PROCESS_CANDIDATE_CLASSES_V543` so a smoke
    run can iterate ``PROCESS_CANDIDATE_IDS_V543`` without a registry lookup
    (P4-C has no registry id — see the module docstring).
    """
    key = str(decoder_id)
    if key not in PROCESS_CANDIDATE_CLASSES_V543:
        raise V543ProcessDecoderError(
            "V543_PROCESS_UNKNOWN_CANDIDATE",
            f"unknown PROCESS candidate {key!r}; this package implements "
            f"{list(PROCESS_CANDIDATE_IDS_V543)} (plan §6.3 :543–547)")
    return PROCESS_CANDIDATE_CLASSES_V543[key](**kwargs)


def process_candidate_report() -> dict[str, Any]:
    """The WP6 parameter table: plan estimate vs measured, with the band check.

    JSON-able; used by the WP6 report and re-checked by
    ``tests/pesn/test_v543_process_decoders.py``.
    """
    from pes2ts_core.pesn.models.v543.registry import (
        PARAM_BAND_MAX_V543,
        PARAM_BAND_MIN_V543,
        PARAM_TARGET_V543,
    )

    rows: list[dict[str, Any]] = []
    for decoder_id in FORMAL_PROCESS_DECODER_IDS_V543:
        measured = int(MEASURED_PARAMS_V543[decoder_id])
        estimate = int(PLAN_PARAM_ESTIMATES_V543[decoder_id])
        rows.append({
            "decoder_id": decoder_id,
            "plan_estimate": estimate,
            "measured": measured,
            "delta_vs_estimate_pct": round(
                100.0 * (measured - estimate) / float(estimate), 2),
            "in_frozen_band": bool(
                PARAM_BAND_MIN_V543 <= measured <= PARAM_BAND_MAX_V543),
            "band": [PARAM_BAND_MIN_V543, PARAM_BAND_MAX_V543],
        })
    measured_control = int(MEASURED_PARAMS_V543[DECODER_ID_P4C])
    rows.append({
        "decoder_id": DECODER_ID_P4C,
        "plan_estimate": None,
        "measured": measured_control,
        "delta_vs_P4_pct": round(
            100.0 * (measured_control - p4_reference_params())
            / float(p4_reference_params()), 2),
        "matched_band": p4c_band(),
        "in_matched_band": bool(
            p4c_band()["min"] <= measured_control <= p4c_band()["max"]),
        "band_rule": "P4 measured ±2 % (plan §6.3 :547), independent of "
                     "the §6.4 36,000 ±5 % band",
    })
    return {
        "track": TRACK_PROCESS,
        "target": PARAM_TARGET_V543,
        "band": [PARAM_BAND_MIN_V543, PARAM_BAND_MAX_V543],
        "candidates": rows,
    }


__all__ = [
    "DECODER_ID_P1",
    "DECODER_ID_P2",
    "DECODER_ID_P3",
    "DECODER_ID_P4",
    "DECODER_ID_P4C",
    "FORMAL_PROCESS_DECODER_IDS_V543",
    "MEASURED_PARAMS_V543",
    "P1_EXPECTED_PARAMS",
    "P2_EXPECTED_PARAMS",
    "P3_EXPECTED_PARAMS",
    "P4C_BAND_PCT",
    "P4C_EXPECTED_PARAMS",
    "P4C_SPEC",
    "P4ControlSpec",
    "P4_EXPECTED_PARAMS",
    "PLAN_PARAM_ESTIMATES_V543",
    "PROCESS_CANDIDATE_CLASSES_V543",
    "PROCESS_CANDIDATE_IDS_V543",
    "PROCESS_INIT_SEED_V543",
    "PROCESS_TOKEN_DIM_V543",
    "ProcessDeepSets",
    "ProcessGraphTransformer",
    "ProcessLatentGraphGNN",
    "ProcessP4Control",
    "ProcessResidualMLP",
    "TokenEdges",
    "TokenTensors",
    "V543ProcessDecoderError",
    "assert_p4c_matches_p4",
    "build_p1",
    "build_p2",
    "build_p3",
    "build_p4",
    "build_p4c",
    "build_process_candidate",
    "deterministic_init",
    "local_token_edges",
    "p4_reference_params",
    "p4c_band",
    "process_candidate_report",
    "process_candidate_specs",
    "register_process_candidates",
    "token_tensors",
]
