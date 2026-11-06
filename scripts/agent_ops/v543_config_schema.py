#!/usr/bin/env python3
"""V5.4.3 dual-Decoder 50K campaign config schema validator (WP0, ADR-0034).

Validates ``config/campaigns/pes2ts_v543_dual_decoder_50k.yaml`` (schema
``pes2ts_v543_dual_decoder_campaign_v1``) and refuses to let a runner start
from an invalid config.  The config is a verbatim transcription of the plan's
appendix A draft (``docs/plans/PES2TS_V5.4.3_双Decoder_50K_全面开发与训练方案_v1.0_20261102.md``)
plus the v542-shaped ``gv`` / ``governance`` / ``budget`` governance blocks;
this module is the WP0 machine gate over it.

Frozen contract highlights
--------------------------
* **schema / identity** — ``schema``/``campaign_id``/``version``/``status``
  exact; the campaign is ``proposed`` at WP0 (an ``executed``/``smoke`` claim
  is a typed rejection).
* **G–V registry** — ``gv`` mirrors ``config/registry/g_v_map.yaml`` entry
  ``V5.4.3``: same ``map_version``, ``primary_g=G3.P.2``, the four supporting
  stages, the frozen 17-gate list (order is part of the freeze) and the
  ``config`` path.  A missing/drifted registry entry is
  ``V543_GV_MISMATCH``.
* **governance** — ``docs_gate`` must be
  ``python scripts/agent_ops/docs_consistency.py --strict``; the validator
  points at this module; the ADR set carries ``ADR-0034``; the authorization
  block records the signed facts (D1–D8 signed 2026-11-02, smoke-gated
  auto-launch, frozen thresholds, ``confirm_authorized=false`` /
  ``sealed_authorized=false`` / ``new_qc_calls=0`` / ``single_lane`` /
  ``swap_allowed=false``); ``t1_start_requires`` is the exact frozen list;
  ``governance.records`` mirrors the frozen record spec (template, instance,
  required fields, ``sha_rule``).
* **source** — the three V542 A0 best checkpoints with EXACT paths + SHA-256
  (plan appendix A); ``freeze_encoder``/``freeze_coarse_move``/
  ``freeze_coarse_process`` true; confirm and new-QC access false.
* **data** — 50,000 pool / 589 dev / 2,000 confirm, confirm
  ``sealed_never_read``, MOVE eligible status ``ok`` only, PROCESS endpoint
  statuses cover all four label statuses.
* **bridge** — ``geometric_latent_bridge_v1`` / ``decoder_context_v1``,
  widths 64/48/48, 8 polar vector channels with channel 0 = ``u0/a_ref``,
  ``parameter_free=true``, no decoder raw xyz, no supervision leak, pair
  direction = canonical key-slot order with signed ``ΔB``, forward edges
  only, 10,000-frame pair audit.
* **tracks** — M0/M1–M3 + conditional M4/M5, P0/P1–P4 + matched control
  P4-C + conditional P5, the frozen V53MP loss names and lambdas
  (0.5/0.02/0.0/0.0 and 0.5/0.5/0.012385049119773344), the endpoint-label
  frame mask for PROCESS.
* **parameter budget (arithmetic, not prose)** — per-track target 36,000,
  band ±5% ⇒ 34,200–37,800, single-decoder hard cap 40,000, whole-model
  target 57,619 + 36,000 + 36,000 = 129,619, in-band maximum
  57,619 + 37,800 + 37,800 = 133,219, global hard cap 150,000.  Every one of
  these is recomputed from the config values; a single drifted number is a
  ``V543_PARAM_BUDGET`` rejection.
* **training dose** — decoder-only, 37,900 flat updates + 400 warmup
  (37,500 post-warmup is derived, never the headline dose), LR 3e-4, AdamW,
  weight decay 1e-4, grad clip 1.0, fp32, batch 8 curves × 4 frames, the
  frozen ``v541_curve_stratified_v1`` spec SHA, inner-val cap 8,192,
  identity-matched resume only, ``torch.compile=false``.
* **cache** — frozen-encoder-only, 200,000 RAM LRU packets, 30 GB disk,
  256 curves/shard, ≥0.80 hit-rate target, joint fine-tune forbidden.
* **screening** — 200-update smoke, 2,000-update diagnostic, seeds 17/29
  then finalist seed 43 with at most 2 finalists per track, H3 selection on
  diagnostics forbidden, the four diagnostic margins.
* **evaluation / statistics** — matched-seed V543 M0/P0 recomputed baseline,
  curve-level cluster bootstrap 10,000 draws with seed root 20261102, Holm
  correction, thresholds frozen before runs, dev one-shot on final frozen
  candidates only, confirm sealed, ``fixed_coverage_direction_at_3`` as the
  new metric, ``q_corr`` diagnostic-only with the ≥3-point / ≥0.70 curve
  validity rules.
* **thresholds (appendix B)** — the 19 signed keys with EXACT values and an
  exact key set (no extra keys, no near-miss relabelling); the
  ``thresholds.signoff`` mirror must be value-identical; the appendix-B
  statistics rows (``bootstrap_draws=10000``, ``bootstrap_seed_root=20261102``)
  are cross-checked against the ``evaluation`` block.
* **resources** — 150,000 whole-model cap, 40,000 per refiner, p95 latency
  ratio 2.0, exactly one Encoder forward per frame, single lane, no swap,
  12 GiB host limit.
* **conditional generation** — Flow NFE {1,2,4,8}, residual-bottleneck
  activation, Diffusion default off, GraphGRU activated by a systematic
  PROCESS failure, no teacher-based sample selection, new signoff required.
* **18-run matrix (appendix C)** — T3 = 7 decoders at seed 17, T4 = 7
  decoders at seed 29, T5 ≤ 4 finalists at seed 43, total 7 + 7 + 4 = 18
  formal runs × 37,900 = 682,200 updates; T6 per-track champions or retain,
  T7 four integration combos ``[B0, BM, BP, BMP]``, T8 separate identity.
  The exact appendix-C run-id set is generated and checked (renamed/extra/
  missing runs are ``V543_RUN_MATRIX``).

Typed rejections (each carries a stable ``code``; 23 codes):

===========================  ==========================================================
code                           rejection
===========================  ==========================================================
``V543_CONFIG_MISSING``      file absent / unreadable / not a mapping
``V543_SCHEMA``              ``schema``/``campaign_id``/``version``/``status`` drift
``V543_MISSING_FIELD``       a required block/field is absent or None
``V543_GOVERNANCE``          ``governance`` block drift (docs gate / validator /
                             ADR-0034 / plan / records spec)
``V543_AUTHORIZATION``       authorization facts drift (confirm/sealed/QC/lane/swap/
                             signed D1–D8 / frozen thresholds / t1_start_requires)
``V543_GV_MISMATCH``         ``gv`` block or registry entry disagrees with
                             ``config/registry/g_v_map.yaml``
``V543_SOURCE``              encoder/seeds/checkpoint path or SHA drift, or frozen-
                             encoder / confirm / new-QC flags wrong
``V543_DATA``                data split/status policy drift
``V543_BRIDGE``              Bridge contract drift (widths/channels/parameter-free/
                             pair direction/leak flags/audit sample)
``V543_TRACK``               track candidate list / loss / lambda / output drift
``V543_PARAM_BUDGET``        parameter arithmetic broken (36K ±5% band, 40K cap,
                             129,619 / 133,219 / 150,000)
``V543_TRAINING``            training dose/optimizer/precision/batch/sampler drift
``V543_CACHE``               cache contract drift
``V543_SCREENING``           screening dose/margins drift
``V543_EVALUATION``          evaluation/statistics contract drift
``V543_THRESHOLD``           signed threshold missing, drifted, extra or null
``V543_RESOURCE``            resource guard drift (single lane / swap / host / latency /
                             encoder-forward count)
``V543_CONDITIONAL``         conditional-generation activation policy drift
``V543_RUN_MATRIX``          18-run matrix / seed assignment / dose drift
``V543_BUDGET_NULL``         ``--formal`` with ``budget.per_candidate_core_time``
                             still null/empty
``V543_BUDGET_INVALID``      ``--formal`` with a malformed budget table
``V543_RECORD_MISSING``      ``--formal`` with the machine-checked governance
                             record instance absent/unreadable/None (incl. a missing
                             ``record_sha256``)
``V543_RECORD_MISMATCH``     a governance record instance (or its config mirror)
                             disagrees with the frozen config — wrong
                             ``config_path``/roles/schema/thresholds or a
                             ``record_sha256`` that does not match the canonical
                             digest
===========================  ==========================================================

Modes
-----
* ``prereg`` (default, also selectable with ``--prereg``) — structural
  validation; a null budget or a missing governance record instance is
  reported as a ``t1_blocked`` reason (WP0 state: the WP7 profile budget and
  the signoff instance land later; the runner never fills defaults).
* ``formal`` — additionally requires every signed threshold non-null, a
  WELL-FORMED budget table (null/empty ⇒ ``V543_BUDGET_NULL``; malformed ⇒
  ``V543_BUDGET_INVALID``) AND the machine-checked governance record instance
  (missing ⇒ ``V543_RECORD_MISSING``, drift ⇒ ``V543_RECORD_MISMATCH``).

``validate_config`` runs these named checks in order (21 checks with the G–V
cross-check enabled — keep this list current):

``schema``, ``campaign_status``, ``required_fields``, ``gv_registry``,
``governance``, ``authorization``, ``records_spec``, ``source``, ``data``,
``bridge``, ``tracks``, ``param_budget``, ``training``, ``cache``,
``screening``, ``evaluation``, ``thresholds``, ``resources``,
``conditional_generation``, ``run_matrix``, plus the ``governance_records``
record gate whose findings land in ``t1_blocked``.

WP0 skeleton boundary (documented, not silent)
----------------------------------------------
Cross-file evidence that cannot exist at WP0 is deliberately NOT asserted
here and is marked with ``TODO(WP1)``/``TODO(WP7)`` comments at the check
site: (a) the three A0 checkpoint files are actually present and their
measured SHA-256 equals the pinned digest, (b) the parity certificate /
``u0/z/q/e0`` zero-residual equations have been measured, (c) the
``budget.per_candidate_core_time`` table has been filled from the same-env
WP7 profile.  Structure, keys, types, enums and every numeric invariant ARE
enforced — only the measured-evidence layer is deferred.

CLI::

    python scripts/agent_ops/v543_config_schema.py                 # prereg mode
    python scripts/agent_ops/v543_config_schema.py --prereg        # explicit
    python scripts/agent_ops/v543_config_schema.py --formal        # formal gate
    python scripts/agent_ops/v543_config_schema.py --config PATH --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import yaml  # PyYAML: existing repo dependency

__all__ = [
    "BAND_HIGH",
    "BAND_LOW",
    "CAMPAIGN_CONFIG_REL",
    "CAMPAIGN_ID",
    "CAMPAIGN_STATUS",
    "DEFAULT_CONFIG",
    "DOSE_FLAT",
    "ENCODER_FROZEN_PARAMS",
    "EXPECTED_RUN_IDS",
    "EXPECTED_RUN_TABLE",
    "FORMAL_DECODER_IDS",
    "FORMAL_RUNS_MAX",
    "FORMAL_STAGE_ORDER",
    "GV_CAMPAIGN_ID",
    "GV_EXPECTED_GATES",
    "KNOWN_DECODERS",
    "PARAM_BAND_PCT",
    "PER_DECODER_CAP",
    "RECORD_SHA_FIELD",
    "RECORD_SHA_RULE",
    "REQUIRED_BLOCKS",
    "SCHEMA_ID",
    "SIGNOFF_RECORD_SCHEMA_ID",
    "SIGNOFF_ROLE_KEYS",
    "SIGNOFF_REQUIRED_ROLES",
    "SINGLE_TRAINABLE_CAP",
    "T1_START_REQUIRES",
    "TARGET_TRACK_PARAMS",
    "THRESHOLDS",
    "TOTAL_FORMAL_UPDATES",
    "TOTAL_MODEL_BAND_MAX",
    "TOTAL_MODEL_HARD_MAX",
    "TOTAL_MODEL_TARGET",
    "TYPED_REJECTIONS",
    "WARMUP_UPDATES",
    "V543SchemaError",
    "load_config",
    "main",
    "validate_config",
]

SCHEMA_ID = "pes2ts_v543_dual_decoder_campaign_v1"
CAMPAIGN_ID = "PES2TS_V543_DUAL_DECODER_50K"
GV_CAMPAIGN_ID = "V5.4.3"
CAMPAIGN_CONFIG_REL = "config/campaigns/pes2ts_v543_dual_decoder_50k.yaml"
PLAN_REL = (
    "docs/plans/PES2TS_V5.4.3_双Decoder_50K_全面开发与训练方案_v1.0_20261102.md"
)
WP_CHECKLIST_REL = "docs/plans/PES2TS_V5.4.3_WP执行清单_v1.0_20261102.md"
ADR_PATH_REL = (
    "docs/design/decisions/ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md"
)
VERSION = "0.1.0-wp0"
#: WP0 state: any executed/smoke claim would be a false completion claim.
CAMPAIGN_STATUS = "proposed"
GV_MAP_REL = "config/registry/g_v_map.yaml"

GV_PRIMARY_G = "G3.P.2"
GV_SUPPORTING_G: Tuple[str, ...] = ("G3.P.1", "G3.P.3", "G0", "G1")
GV_EXPECTED_GATES: Tuple[str, ...] = (
    "V543-WP0-prereg",
    "V543-WP1-encoder-parity",
    "V543-WP2-bridge-context",
    "V543-WP3-sufficiency",
    "V543-WP4-registry-trainer",
    "V543-smoke-200",
    "V543-profile",
    "V543-diagnostic-2000",
    "V543-H0-legality",
    "V543-H1-resource",
    "V543-H2-move",
    "V543-H2-process",
    "V543-H3-move",
    "V543-H3-process",
    "V543-budget-signoff",
    "V543-formal",
    "V543-S2-conditional",
)

#: The 23 typed rejection codes above — the authoritative count for docs.
TYPED_REJECTIONS: Tuple[str, ...] = (
    "V543_CONFIG_MISSING",
    "V543_SCHEMA",
    "V543_MISSING_FIELD",
    "V543_GOVERNANCE",
    "V543_AUTHORIZATION",
    "V543_GV_MISMATCH",
    "V543_SOURCE",
    "V543_DATA",
    "V543_BRIDGE",
    "V543_TRACK",
    "V543_PARAM_BUDGET",
    "V543_TRAINING",
    "V543_CACHE",
    "V543_SCREENING",
    "V543_EVALUATION",
    "V543_THRESHOLD",
    "V543_RESOURCE",
    "V543_CONDITIONAL",
    "V543_RUN_MATRIX",
    "V543_BUDGET_NULL",
    "V543_BUDGET_INVALID",
    "V543_RECORD_MISSING",
    "V543_RECORD_MISMATCH",
)

REQUIRED_BLOCKS: Tuple[str, ...] = (
    "schema",
    "campaign_id",
    "version",
    "status",
    "gv",
    "governance",
    "source",
    "data",
    "bridge",
    "tracks",
    "training",
    "cache",
    "screening",
    "evaluation",
    "thresholds",
    "resources",
    "conditional_generation",
    "run_matrix",
    "budget",
)

# ---------------------------------------------------------------------------
# Frozen parameter arithmetic (plan §6.4 / §8.3 / ADR-0034 §4).
# ---------------------------------------------------------------------------
#: A0 encoder + D0 + P0 frozen component count (V542 joint invariant 57,619).
ENCODER_FROZEN_PARAMS = 57619
TARGET_TRACK_PARAMS = 36000
PARAM_BAND_PCT = 5.0
BAND_LOW = 34200            # 36,000 - 5%
BAND_HIGH = 37800           # 36,000 + 5%
PER_DECODER_CAP = 40000     # single-candidate hard cap
SINGLE_TRAINABLE_CAP = 40000
TOTAL_MODEL_TARGET = 129619     # 57,619 + 36,000 + 36,000
TOTAL_MODEL_BAND_MAX = 133219   # 57,619 + 37,800 + 37,800
TOTAL_MODEL_HARD_MAX = 150000

# ---------------------------------------------------------------------------
# Frozen training dose (plan §7.2 / §8.1; ADR-0034 §4).
# 37,900 flat updates + 400 warmup; the V542 400+37,500 two-phase split is
# explicitly retired (§2 difference 6) — 37,500 is derived, never a headline.
# ---------------------------------------------------------------------------
DOSE_FLAT = 37900
WARMUP_UPDATES = 400
DERIVED_POST_WARMUP = DOSE_FLAT - WARMUP_UPDATES   # 37,500 (derived only)
SAMPLER_ID = "v541_curve_stratified_v1"
SAMPLER_SPEC_SHA256 = (
    "0769817ef01dca3abbb1d67f5405e98138b4707e74b690240d3f9b6821b65d9e"
)

# ---------------------------------------------------------------------------
# 18-run formal matrix (plan §8.1 / appendix C).
# ---------------------------------------------------------------------------
FORMAL_RUNS_MAX = 18
T3_SEED = 17
T4_SEED = 29
T5_SEED = 43
T5_MAX_RUNS = 4
MOVE_FORMAL = ("M1", "M2", "M3")
PROCESS_FORMAL = ("P1", "P2", "P3", "P4")
FORMAL_DECODER_IDS: Tuple[str, ...] = MOVE_FORMAL + PROCESS_FORMAL   # 7
T3_RUN_DECODERS: Tuple[str, ...] = FORMAL_DECODER_IDS
T4_RUN_DECODERS: Tuple[str, ...] = FORMAL_DECODER_IDS
#: Formal-run total dose: 18 × 37,900 = 682,200 updates (§8.1).
TOTAL_FORMAL_UPDATES = FORMAL_RUNS_MAX * DOSE_FLAT
#: Frozen formal stage order (T0 parity → T3 → T4 → T5 → T6/T7).
FORMAL_STAGE_ORDER: Tuple[str, ...] = (
    "T0_M0_P0_parity_three_seeds",
    "T3_seven_decoders_seed17",
    "T4_seven_decoders_seed29",
    "T5_two_finalists_per_track_seed43",
    "T6_two_track_champions_or_retain",
    "T7_integration_B0_BM_BP_BMP",
    "T8_conditional_separate_identity",
)

#: Whole decoder registry (baselines, formal candidates, matched control,
#: conditional).  Only ``FORMAL_DECODER_IDS`` ever enter the 18-run matrix.
MOVE_DECODERS: Tuple[str, ...] = ("M0", "M1", "M2", "M3", "M4", "M5")
PROCESS_DECODERS: Tuple[str, ...] = ("P0", "P1", "P2", "P3", "P4", "P5")
KNOWN_DECODERS = frozenset(MOVE_DECODERS + PROCESS_DECODERS + ("P4-C",))

#: Appendix-C run table: run_id -> (track, decoder, seed, stage).
EXPECTED_RUN_TABLE: Dict[str, Dict[str, Any]] = {}
for _decoder in T3_RUN_DECODERS:
    EXPECTED_RUN_TABLE[f"V543_{_decoder}_S{T3_SEED}"] = {
        "track": "MOVE" if _decoder.startswith("M") else "PROCESS",
        "decoder": _decoder,
        "seed": T3_SEED,
        "stage": "T3",
        "dose": DOSE_FLAT,
    }
for _decoder in T4_RUN_DECODERS:
    EXPECTED_RUN_TABLE[f"V543_{_decoder}_S{T4_SEED}"] = {
        "track": "MOVE" if _decoder.startswith("M") else "PROCESS",
        "decoder": _decoder,
        "seed": T4_SEED,
        "stage": "T4",
        "dose": DOSE_FLAT,
    }
#: T5 rows carry the frozen placeholder form from appendix C — the concrete
#: decoder is only known after the T3/T4 screen.
for _placeholder in (
    "V543_<MOVE_top1>_S43",
    "V543_<MOVE_top2>_S43",
    "V543_<PROC_top1>_S43",
    "V543_<PROC_top2>_S43",
):
    EXPECTED_RUN_TABLE[_placeholder] = {
        "track": "MOVE" if "MOVE" in _placeholder else "PROCESS",
        "decoder": "<top1>" if "top1" in _placeholder else "<top2>",
        "seed": T5_SEED,
        "stage": "T5",
        "dose": DOSE_FLAT,
    }
EXPECTED_RUN_IDS = frozenset(EXPECTED_RUN_TABLE)
EXPECTED_RUN_COUNT = 18

# ---------------------------------------------------------------------------
# Appendix B signed thresholds — exact values, exact key set.
# ---------------------------------------------------------------------------
THRESHOLDS: Dict[str, float] = {
    "move_rmse_regression_rel_max_pct": 2.0,
    "move_sign_drop_abs_max": 0.01,
    "move_dead_increase_abs_max": 0.01,
    "move_fixed_coverage_regression_rel_max_pct": 2.0,
    "move_h3_rmse_rel_min_pct": 3.0,
    "move_h3_direction_rel_min_pct": 3.0,
    "move_h3_direction_abs_min": 0.01,
    "process_auprc_drop_abs_max": 0.01,
    "process_recall_at_fpr_drop_abs_max": 0.02,
    "process_q_mae_regression_rel_max_pct": 5.0,
    "process_ordering_drop_abs_max": 0.02,
    "process_ece_increase_abs_max": 0.02,
    "process_h3_auprc_abs_min": 0.02,
    "process_h3_q_mae_rel_min_pct": 5.0,
    "process_fpr_constraint": 0.05,
    "move_exit_alpha_each": 0.025,
    "process_family_alpha": 0.025,
    "max_core_time_ratio": 2.0,
    "max_latency_ratio_p95": 2.0,
}
#: Appendix-B statistics rows live in the ``evaluation`` block.
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED_ROOT = 20261102

# ---------------------------------------------------------------------------
# T1 start requirements (frozen order; gates then record requirements).
# ---------------------------------------------------------------------------
T1_CANONICAL_GATES: Tuple[str, ...] = (
    "V543-WP1-encoder-parity",
    "V543-WP2-bridge-context",
    "V543-WP3-sufficiency",
    "V543-WP4-registry-trainer",
    "V543-smoke-200",
    "V543-profile",
    "V543-diagnostic-2000",
    "V543-H0-legality",
    "V543-H1-resource",
    "V543-budget-signoff",
)
T1_RECORD_REQUIREMENTS: Tuple[str, ...] = (
    "runplan_pin_record",
    "preregistration_signoff_record",
)
T1_START_REQUIRES: Tuple[str, ...] = T1_CANONICAL_GATES + T1_RECORD_REQUIREMENTS

#: Frozen source checkpoints (plan appendix A; same-seed V542 A0 best).
SOURCE_CHECKPOINTS: Dict[str, Dict[str, str]] = {
    "17": {
        "path": "outputs/pesn/v542_backbone_50k/runs/V542_A0_S17/checkpoint_best.pt",
        "sha256": "cab512b0094788093a9969484db7f9f303802d2fff0d7380bffb8923fc7f89bb",
    },
    "29": {
        "path": "outputs/pesn/v542_backbone_50k/runs/V542_A0_S29/checkpoint_best.pt",
        "sha256": "c52db93253777a00637ea6cb41eb1366eeeaf37464d38a3d42c04692f496466b",
    },
    "43": {
        "path": "outputs/pesn/v542_backbone_50k/runs/V542_A0_S43/checkpoint_best.pt",
        "sha256": "dd37c7e997d74c6426db9052918a2400687ce827600fc3dedd69efd01b6be8d4",
    },
}

# ---------------------------------------------------------------------------
# Governance records (frozen spec; validator mirrors it and checks the instance).
# ---------------------------------------------------------------------------
RECORD_SHA_RULE = "sha256_of_canonical_file_bytes"
RECORD_SHA_FIELD = "record_sha256"
PREREG_SIGNOFF_TEMPLATE = (
    "config/campaigns/v543_templates/preregistration_signoff.template.json"
)
PREREG_SIGNOFF_INSTANCE = (
    "outputs/pesn/v543_dual_decoder_50k/plan/preregistration_signoff.json"
)
RUNPLAN_PIN_TEMPLATE = (
    "config/campaigns/v543_templates/runplan_pin.template.json"
)
RUNPLAN_PIN_INSTANCE = (
    "outputs/pesn/v543_dual_decoder_50k/plan/runplan.pin.json"
)
SIGNOFF_RECORD_SCHEMA_ID = "pes2ts_v543_preregistration_signoff_v1"
#: Reviewer is NEVER a signatory; statistics/evaluation belong to the owner.
SIGNOFF_REQUIRED_ROLES: Dict[str, str] = {
    "statistics_owner": "main_session_owner",
    "evaluation_signoff": "main_session_owner",
    "oracle_reviewer": "oracle_gate0_reviewer",
}
SIGNOFF_ROLE_KEYS = frozenset({
    "authorization", "governance_owner", "statistics_owner",
    "governance_signoff", "parity_signoff", "training_signoff",
    "evaluation_signoff", "resource_signoff", "oracle_reviewer",
    "formal_launch_note",
})
SIGNATORY_ROLES_NO_ORACLE = frozenset({
    "authorization", "governance_owner", "statistics_owner",
    "governance_signoff", "parity_signoff", "training_signoff",
    "evaluation_signoff", "resource_signoff",
})
GOVERNANCE_RECORDS: Dict[str, Dict[str, Any]] = {
    "runplan_pin_record": {
        "template": RUNPLAN_PIN_TEMPLATE,
        "instance": RUNPLAN_PIN_INSTANCE,
        "required_fields": (
            "campaign_id", "config_path", "schema_id", "seeds",
            "runplan_sha256", "frozen_on", "frozen_by",
        ),
        "sha_rule": RECORD_SHA_RULE,
        "machine_checked": False,
    },
    "preregistration_signoff_record": {
        "template": PREREG_SIGNOFF_TEMPLATE,
        "instance": PREREG_SIGNOFF_INSTANCE,
        "required_fields": (
            "campaign_id", "config_path", "schema", "schema_id",
            "gv_campaign_id", "thresholds_signed_frozen", "signed_by_roles",
            RECORD_SHA_FIELD,
        ),
        "sha_rule": RECORD_SHA_RULE,
        "machine_checked": True,
        "expected": {
            "campaign_id": CAMPAIGN_ID,
            "schema_id": SCHEMA_ID,
            "gv_campaign_id": GV_CAMPAIGN_ID,
            "config_path": CAMPAIGN_CONFIG_REL,
            "schema": SIGNOFF_RECORD_SCHEMA_ID,
        },
        "threshold_mirror": "thresholds_signed_frozen",
        "signoff_roles": True,
    },
}

_DEFAULT_CONFIG = (
    PROJECT_ROOT / "config" / "campaigns" / "pes2ts_v543_dual_decoder_50k.yaml"
)
DEFAULT_CONFIG = _DEFAULT_CONFIG

_SHA64_RE = re.compile(r"^[0-9a-f]{64}$")


class V543SchemaError(ValueError):
    """Typed rejection with a stable machine code."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# loading / helpers
# ---------------------------------------------------------------------------
def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Read + minimally check the V543 campaign yaml (fail-closed)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not cfg_path.is_file():
        raise V543SchemaError("V543_CONFIG_MISSING", str(cfg_path))
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V543SchemaError("V543_CONFIG_MISSING", f"{cfg_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise V543SchemaError("V543_CONFIG_MISSING", f"{cfg_path}: not a mapping")
    return raw


def _err(code: str, detail: str) -> None:
    raise V543SchemaError(code, detail)


def _canonical_json_bytes(obj: Any) -> bytes:
    """Compact sorted-key JSON, UTF-8, never escaped.

    This is the ONE frozen canonical recipe the run-identity layer uses
    (``canonical_bytes`` in the identity module).  V5.4.3 has no identity
    module yet (WP1 deliverable), so the recipe is re-declared here from the
    frozen V542 definition instead of imported; TODO(WP1): import
    ``pes2ts_core.pesn.models.v543.identity.canonical_bytes`` read-only once
    it exists and delete this local copy (byte-identical by construction).
    """
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _record_canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Canonical record digest: sha256 over the payload minus its own digest."""
    body = {k: v for k, v in payload.items() if k != RECORD_SHA_FIELD}
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def _block(cfg: Mapping[str, Any], name: str, code: str = "V543_MISSING_FIELD"
           ) -> Mapping[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        _err(code, f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _num(block: Mapping[str, Any], key: str, expected: float, code: str,
         where: str) -> None:
    """Numeric equality that refuses bools and token/near-miss coercion."""
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _err(code, f"{where}.{key} must be the number {expected}, got {value!r}")
    if float(value) != float(expected):
        _err(code, f"{where}.{key} must be {expected}, got {value!r}")


def _ival(block: Mapping[str, Any], key: str, expected: int, code: str,
          where: str) -> None:
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        _err(code, f"{where}.{key} must be the integer {expected}, "
                   f"got {value!r}")
    if value != expected:
        _err(code, f"{where}.{key} must be {expected}, got {value!r}")


def _flag(block: Mapping[str, Any], key: str, expected: bool, code: str,
          where: str) -> None:
    value = block.get(key)
    if value is not expected:
        _err(code, f"{where}.{key} must be exactly the boolean {expected}, "
                   f"got {value!r} ({type(value).__name__})")


def _str(block: Mapping[str, Any], key: str, expected: str, code: str,
         where: str) -> None:
    value = block.get(key)
    if value != expected:
        _err(code, f"{where}.{key} must be {expected!r}, got {value!r}")


def _seq(block: Mapping[str, Any], key: str, expected: Sequence[Any],
         code: str, where: str) -> None:
    value = block.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _err(code, f"{where}.{key} must be a list, got {value!r}")
    if list(value) != list(expected):
        _err(code, f"{where}.{key} must be exactly {list(expected)}, "
                   f"got {list(value)!r}")


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA64_RE.match(value))


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------
def _check_schema(cfg: Mapping[str, Any]) -> None:
    if cfg.get("schema") != SCHEMA_ID:
        _err("V543_SCHEMA", f"expected {SCHEMA_ID!r}, got {cfg.get('schema')!r}")
    if cfg.get("campaign_id") != CAMPAIGN_ID:
        _err("V543_SCHEMA",
             f"campaign_id expected {CAMPAIGN_ID!r}, "
             f"got {cfg.get('campaign_id')!r}")
    if cfg.get("version") != VERSION:
        _err("V543_SCHEMA",
             f"version must be the WP0 freeze {VERSION!r}, "
             f"got {cfg.get('version')!r}")


def _check_campaign_status(cfg: Mapping[str, Any]) -> None:
    """WP0 must stay ``proposed`` — no smoke/executed completion claim."""
    if cfg.get("status") != CAMPAIGN_STATUS:
        _err(
            "V543_SCHEMA",
            f"status must be {CAMPAIGN_STATUS!r} at WP0 (registry V5.4.3 is "
            f"proposed; executed/smoke_passed claims are typed rejections), "
            f"got {cfg.get('status')!r}",
        )


def _check_required_fields(cfg: Mapping[str, Any]) -> None:
    missing = [k for k in REQUIRED_BLOCKS if cfg.get(k) is None]
    if missing:
        _err("V543_MISSING_FIELD", f"missing/None required block(s): {missing}")


def _load_gv_registry() -> Dict[str, Any]:
    map_path = PROJECT_ROOT / GV_MAP_REL
    try:
        raw = yaml.safe_load(map_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        _err("V543_GV_MISMATCH", f"cannot read {GV_MAP_REL}: {exc}")
    if not isinstance(raw, Mapping):
        _err("V543_GV_MISMATCH", f"{GV_MAP_REL} is not a mapping")
    return dict(raw)


def _check_gv_block(cfg: Mapping[str, Any]) -> None:
    gv = _block(cfg, "gv", "V543_GV_MISMATCH")
    _str(gv, "campaign_id", GV_CAMPAIGN_ID, "V543_GV_MISMATCH", "gv")
    _str(gv, "primary_g", GV_PRIMARY_G, "V543_GV_MISMATCH", "gv")
    _seq(gv, "supporting_g", GV_SUPPORTING_G, "V543_GV_MISMATCH", "gv")
    _seq(gv, "gates", GV_EXPECTED_GATES, "V543_GV_MISMATCH", "gv")

    document = _load_gv_registry()
    map_version = document.get("map_version")
    if gv.get("map_version") != map_version:
        _err(
            "V543_GV_MISMATCH",
            f"gv.map_version must equal the live registry map_version "
            f"{map_version!r}, got {gv.get('map_version')!r}",
        )
    campaigns = document.get("campaigns")
    if not isinstance(campaigns, Mapping):
        _err("V543_GV_MISMATCH", f"{GV_MAP_REL} carries no campaigns mapping")
    entry = campaigns.get(GV_CAMPAIGN_ID)
    if not isinstance(entry, Mapping):
        _err(
            "V543_GV_MISMATCH",
            f"registry entry {GV_CAMPAIGN_ID!r} missing from {GV_MAP_REL} "
            "(WP0 registers V5.4.3 as proposed)",
        )
    if entry.get("status") != "proposed":
        _err(
            "V543_GV_MISMATCH",
            f"registry {GV_CAMPAIGN_ID} status must be 'proposed' at WP0, "
            f"got {entry.get('status')!r}",
        )
    if entry.get("primary_g") != GV_PRIMARY_G:
        _err("V543_GV_MISMATCH",
             f"registry {GV_CAMPAIGN_ID} primary_g must be {GV_PRIMARY_G!r}, "
             f"got {entry.get('primary_g')!r}")
    entry_supporting = tuple(entry.get("supporting_g") or ())
    if set(entry_supporting) != set(GV_SUPPORTING_G):
        _err(
            "V543_GV_MISMATCH",
            f"registry {GV_CAMPAIGN_ID} supporting_g {list(entry_supporting)} "
            f"!= config {list(GV_SUPPORTING_G)}",
        )
    entry_gates = tuple(entry.get("gates") or ())
    if set(entry_gates) != set(GV_EXPECTED_GATES):
        _err(
            "V543_GV_MISMATCH",
            f"registry {GV_CAMPAIGN_ID} gates must declare exactly the frozen "
            f"gate set; missing="
            f"{sorted(set(GV_EXPECTED_GATES) - set(entry_gates))} "
            f"extra={sorted(set(entry_gates) - set(GV_EXPECTED_GATES))}",
        )
    if entry.get("config") != CAMPAIGN_CONFIG_REL:
        _err(
            "V543_GV_MISMATCH",
            f"registry {GV_CAMPAIGN_ID}.config must point at "
            f"{CAMPAIGN_CONFIG_REL!r}, got {entry.get('config')!r}",
        )


def _check_governance(cfg: Mapping[str, Any]) -> None:
    gov = _block(cfg, "governance")
    docs_gate = str(gov.get("docs_gate") or "")
    if "scripts/agent_ops/docs_consistency.py" not in docs_gate \
            or "--strict" not in docs_gate:
        _err(
            "V543_GOVERNANCE",
            "governance.docs_gate must be "
            "'python scripts/agent_ops/docs_consistency.py --strict', "
            f"got {docs_gate!r}",
        )
    validator = str(gov.get("schema_validator") or "")
    if "scripts/agent_ops/v543_config_schema.py" not in validator:
        _err(
            "V543_GOVERNANCE",
            "governance.schema_validator must point at "
            f"scripts/agent_ops/v543_config_schema.py, got {validator!r}",
        )
    if CAMPAIGN_CONFIG_REL not in validator:
        _err(
            "V543_GOVERNANCE",
            "governance.schema_validator must name the frozen campaign config "
            f"{CAMPAIGN_CONFIG_REL!r}, got {validator!r}",
        )
    adrs = tuple(gov.get("adrs") or ())
    if "ADR-0034" not in adrs:
        _err(
            "V543_GOVERNANCE",
            "governance.adrs must include ADR-0034 (the single V5.4.3 "
            f"governance anchor), got {list(adrs)}",
        )
    if gov.get("plan") != PLAN_REL:
        _err("V543_GOVERNANCE",
             f"governance.plan must be {PLAN_REL!r}, got {gov.get('plan')!r}")
    if gov.get("wp_checklist") != WP_CHECKLIST_REL:
        _err("V543_GOVERNANCE",
             f"governance.wp_checklist must be {WP_CHECKLIST_REL!r}, "
             f"got {gov.get('wp_checklist')!r}")
    if gov.get("adr_path") != ADR_PATH_REL:
        _err("V543_GOVERNANCE",
             f"governance.adr_path must be {ADR_PATH_REL!r}, "
             f"got {gov.get('adr_path')!r}")
    constitution = str(gov.get("constitution") or "")
    if "开发宪法" not in constitution:
        _err("V543_GOVERNANCE",
             f"governance.constitution must name the dev constitution, "
             f"got {constitution!r}")
    contracts = tuple(gov.get("contracts") or ())
    for required in (
        "docs/contracts/PES2TS_MoveProcessContract_v1.md",
        "docs/contracts/PES2TS_ReactionIntent_Contract_v2.md",
        "docs/contracts/PES2TS_GeometryPolicyContract_v2.md",
    ):
        if required not in contracts:
            _err("V543_GOVERNANCE",
                 f"governance.contracts missing {required!r}; got "
                 f"{list(contracts)}")


def _check_authorization(cfg: Mapping[str, Any]) -> None:
    gov = _block(cfg, "governance")
    auth = gov.get("authorization")
    if not isinstance(auth, Mapping):
        _err("V543_MISSING_FIELD", "governance.authorization must be a mapping")
    _flag(auth, "implementation_authorized", True, "V543_AUTHORIZATION",
          "governance.authorization")
    _flag(auth, "formal_training_authorized", True, "V543_AUTHORIZATION",
          "governance.authorization")
    _flag(auth, "thresholds_signed_frozen", True, "V543_AUTHORIZATION",
          "governance.authorization")
    _flag(auth, "confirm_authorized", False, "V543_AUTHORIZATION",
          "governance.authorization")
    _flag(auth, "sealed_authorized", False, "V543_AUTHORIZATION",
          "governance.authorization")
    _flag(auth, "single_lane", True, "V543_AUTHORIZATION",
          "governance.authorization")
    _flag(auth, "swap_allowed", False, "V543_AUTHORIZATION",
          "governance.authorization")
    _ival(auth, "new_qc_calls", 0, "V543_AUTHORIZATION",
          "governance.authorization")
    _str(auth, "signed_on", "2026-11-02", "V543_AUTHORIZATION",
         "governance.authorization")
    decisions = tuple(auth.get("signed_decisions") or ())
    if decisions != tuple(f"D{i}" for i in range(1, 9)):
        _err(
            "V543_AUTHORIZATION",
            "governance.authorization.signed_decisions must be the signed "
            f"[D1..D8], got {list(decisions)}",
        )
    run_auth = tuple(auth.get("run_authorizations") or ())
    for required in (
        "autonomous_go_no_go_within_governance_gates",
        "smoke_gated_auto_launch_formal",
        "confirm_sealed_qc_zero_dev_t7_one_shot",
    ):
        if required not in run_auth:
            _err(
                "V543_AUTHORIZATION",
                f"governance.authorization.run_authorizations missing "
                f"{required!r}; got {list(run_auth)}",
            )
    _seq(auth, "t1_start_requires", T1_START_REQUIRES, "V543_AUTHORIZATION",
         "governance.authorization")
    # every non-record T1 requirement must be a declared registry gate
    undeclared = [
        g for g in T1_CANONICAL_GATES if g not in GV_EXPECTED_GATES
    ]
    if undeclared:
        _err(
            "V543_AUTHORIZATION",
            f"t1_start_requires names gate(s) {undeclared} that the frozen "
            "gate list does not declare",
        )


def _check_records_spec(cfg: Mapping[str, Any]) -> None:
    gov = _block(cfg, "governance")
    records = gov.get("records")
    if not isinstance(records, Mapping):
        _err("V543_GOVERNANCE", "governance.records must be a mapping")
    for rid, spec in GOVERNANCE_RECORDS.items():
        got = records.get(rid)
        if not isinstance(got, Mapping):
            _err("V543_GOVERNANCE", f"governance.records.{rid} missing")
        for key in ("template", "instance", "required_fields", "sha_rule"):
            if got.get(key) is None:
                _err("V543_GOVERNANCE",
                     f"governance.records.{rid}.{key} missing")
        if got.get("template") != spec["template"]:
            _err("V543_GOVERNANCE",
                 f"governance.records.{rid}.template must be "
                 f"{spec['template']!r}, got {got.get('template')!r}")
        if got.get("instance") != spec["instance"]:
            _err("V543_GOVERNANCE",
                 f"governance.records.{rid}.instance must be "
                 f"{spec['instance']!r}, got {got.get('instance')!r}")
        if tuple(got.get("required_fields") or ()) != tuple(
                spec["required_fields"]):
            _err(
                "V543_GOVERNANCE",
                f"governance.records.{rid}.required_fields must be "
                f"{list(spec['required_fields'])}, "
                f"got {list(got.get('required_fields') or ())}",
            )
        if got.get("sha_rule") != RECORD_SHA_RULE:
            _err("V543_GOVERNANCE",
                 f"governance.records.{rid}.sha_rule must be "
                 f"{RECORD_SHA_RULE!r}, got {got.get('sha_rule')!r}")
    signoff = records.get("preregistration_signoff_record")
    if isinstance(signoff, Mapping):
        expected = GOVERNANCE_RECORDS["preregistration_signoff_record"]["expected"]
        for key, value in expected.items():
            if (signoff.get("expected") or {}).get(key) != value:
                _err(
                    "V543_GOVERNANCE",
                    f"governance.records.preregistration_signoff_record."
                    f"expected.{key} must be {value!r}, got "
                    f"{(signoff.get('expected') or {}).get(key)!r}",
                )


def _check_source(cfg: Mapping[str, Any]) -> None:
    src = _block(cfg, "source", "V543_SOURCE")
    _str(src, "encoder", "V542_A0", "V543_SOURCE", "source")
    _seq(src, "seeds", [17, 29, 43], "V543_SOURCE", "source")
    _str(src, "checkpoint_rule", "same_seed_v542_a0_best", "V543_SOURCE",
         "source")
    for flag in ("freeze_encoder", "freeze_coarse_move",
                 "freeze_coarse_process"):
        _flag(src, flag, True, "V543_SOURCE", "source")
    _flag(src, "allow_confirm_access", False, "V543_SOURCE", "source")
    _flag(src, "allow_new_qc_calls", False, "V543_SOURCE", "source")
    checkpoints = src.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        _err("V543_SOURCE", "source.checkpoints must be a mapping")
    if set(checkpoints) != set(SOURCE_CHECKPOINTS):
        _err(
            "V543_SOURCE",
            f"source.checkpoints must cover exactly seeds "
            f"{sorted(SOURCE_CHECKPOINTS)}, got {sorted(checkpoints)}",
        )
    for seed, expected in SOURCE_CHECKPOINTS.items():
        entry = checkpoints[seed]
        if not isinstance(entry, Mapping):
            _err("V543_SOURCE", f"source.checkpoints.{seed} must be a mapping")
        if entry.get("path") != expected["path"]:
            _err(
                "V543_SOURCE",
                f"source.checkpoints.{seed}.path must be "
                f"{expected['path']!r}, got {entry.get('path')!r}",
            )
        if entry.get("sha256") != expected["sha256"]:
            _err(
                "V543_SOURCE",
                f"source.checkpoints.{seed}.sha256 must be the pinned digest "
                f"{expected['sha256']!r}, got {entry.get('sha256')!r}",
            )
        if not _sha256(entry.get("sha256")):
            _err("V543_SOURCE",
                 f"source.checkpoints.{seed}.sha256 is not a sha256 digest")
    # TODO(WP1): measure the three files on disk and compare against the
    # pinned digest + prove the seven zero-residual equations (plan §5.8 /
    # §7.1 S0-A).  WP0 has no artifact to read yet, so only the frozen
    # identity is asserted here.


def _check_data(cfg: Mapping[str, Any]) -> None:
    data = _block(cfg, "data", "V543_DATA")
    _str(data, "tables_dir", "/pes2ts_data/v53mp_50k_data", "V543_DATA",
         "data")
    _str(data, "split_manifest",
         "/pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json", "V543_DATA",
         "data")
    _ival(data, "pool", 50000, "V543_DATA", "data")
    _ival(data, "dev", 589, "V543_DATA", "data")
    _ival(data, "confirm", 2000, "V543_DATA", "data")
    _str(data, "confirm_policy", "sealed_never_read", "V543_DATA", "data")
    _seq(data, "label_statuses", ["ok", "masked", "terminal", "conflict"],
         "V543_DATA", "data")
    _str(data, "move_eligible_status", "ok", "V543_DATA", "data")
    _seq(data, "process_endpoint_statuses",
         ["ok", "masked", "terminal", "conflict"], "V543_DATA", "data")


def _check_bridge(cfg: Mapping[str, Any]) -> None:
    bridge = _block(cfg, "bridge", "V543_BRIDGE")
    _str(bridge, "schema", "geometric_latent_bridge_v1", "V543_BRIDGE", "bridge")
    _str(bridge, "context_schema", "decoder_context_v1", "V543_BRIDGE",
         "bridge")
    _ival(bridge, "graph_scalar_dim", 64, "V543_BRIDGE", "bridge")
    _ival(bridge, "key_scalar_dim", 48, "V543_BRIDGE", "bridge")
    _ival(bridge, "pair_scalar_dim", 48, "V543_BRIDGE", "bridge")
    _ival(bridge, "vector_channels", 8, "V543_BRIDGE", "bridge")
    _str(bridge, "vector_type", "polar", "V543_BRIDGE", "bridge")
    _ival(bridge, "reference_move_channel", 0, "V543_BRIDGE", "bridge")
    _flag(bridge, "parameter_free", True, "V543_BRIDGE", "bridge")
    _flag(bridge, "allow_decoder_raw_xyz", False, "V543_BRIDGE", "bridge")
    _flag(bridge, "allow_supervision_leak", False, "V543_BRIDGE", "bridge")
    _str(bridge, "pair_direction",
         "canonical_key_slot_order_with_signed_delta_b", "V543_BRIDGE",
         "bridge")
    _str(bridge, "pair_source", "forward_edges_only", "V543_BRIDGE", "bridge")
    _ival(bridge, "audit_pair_equivalence_sample", 10000, "V543_BRIDGE",
          "bridge")


def _check_tracks(cfg: Mapping[str, Any]) -> None:
    tracks = _block(cfg, "tracks", "V543_TRACK")
    move = tracks.get("move")
    process = tracks.get("process")
    if not isinstance(move, Mapping):
        _err("V543_TRACK", "tracks.move must be a mapping")
    if not isinstance(process, Mapping):
        _err("V543_TRACK", "tracks.process must be a mapping")
    _str(move, "baseline", "M0", "V543_TRACK", "tracks.move")
    _seq(move, "formal_candidates", list(MOVE_FORMAL), "V543_TRACK",
         "tracks.move")
    _seq(move, "conditional_candidates", ["M4", "M5"], "V543_TRACK",
         "tracks.move")
    _str(move, "loss", "frozen_v53mp_move_atom_pair_cos", "V543_TRACK",
         "tracks.move")
    _num(move, "lambda_pair", 0.5, "V543_TRACK", "tracks.move")
    _num(move, "lambda_cos", 0.02, "V543_TRACK", "tracks.move")
    _num(move, "lambda_strong", 0.0, "V543_TRACK", "tracks.move")
    _num(move, "lambda_dead", 0.0, "V543_TRACK", "tracks.move")
    _ival(move, "max_trainable_params", PER_DECODER_CAP, "V543_TRACK",
          "tracks.move")
    _ival(move, "target_params", TARGET_TRACK_PARAMS, "V543_TRACK",
          "tracks.move")
    _num(move, "param_band_pct", PARAM_BAND_PCT, "V543_TRACK", "tracks.move")
    _str(move, "output", "residual_key_displacement", "V543_TRACK",
         "tracks.move")

    _str(process, "baseline", "P0", "V543_TRACK", "tracks.process")
    _seq(process, "formal_candidates", list(PROCESS_FORMAL), "V543_TRACK",
         "tracks.process")
    _str(process, "matched_control", "P4-C", "V543_TRACK", "tracks.process")
    _seq(process, "conditional_candidates", ["P5"], "V543_TRACK",
         "tracks.process")
    _str(process, "loss", "frozen_v53mp_process", "V543_TRACK",
         "tracks.process")
    _num(process, "lambda_rank", 0.5, "V543_TRACK", "tracks.process")
    _num(process, "lambda_q", 0.5, "V543_TRACK", "tracks.process")
    _num(process, "lambda_endpoint", 0.012385049119773344, "V543_TRACK",
         "tracks.process")
    _ival(process, "max_trainable_params", PER_DECODER_CAP, "V543_TRACK",
          "tracks.process")
    _ival(process, "target_params", TARGET_TRACK_PARAMS, "V543_TRACK",
          "tracks.process")
    _num(process, "param_band_pct", PARAM_BAND_PCT, "V543_TRACK",
         "tracks.process")
    _str(process, "output", "residual_q_raw_endpoint_logit", "V543_TRACK",
         "tracks.process")
    _flag(process, "share_order_with_q", True, "V543_TRACK", "tracks.process")
    _str(process, "process_frame_mask", "endpoint_label_mask_all_statuses",
         "V543_TRACK", "tracks.process")


def _check_param_budget(cfg: Mapping[str, Any]) -> None:
    """Recompute the whole parameter arithmetic from the config values."""
    tracks = _block(cfg, "tracks", "V543_PARAM_BUDGET")
    move = tracks.get("move") or {}
    process = tracks.get("process") or {}
    resources = _block(cfg, "resources", "V543_PARAM_BUDGET")
    move_target = move.get("target_params")
    process_target = process.get("target_params")
    band_pct = move.get("param_band_pct")
    if move_target != process_target:
        _err(
            "V543_PARAM_BUDGET",
            "both tracks must share the same 36,000 parameter target "
            f"(capacity matching), got move={move_target!r} "
            f"process={process_target!r}",
        )
    if band_pct != PARAM_BAND_PCT:
        _err("V543_PARAM_BUDGET", f"param band must be ±5%, got {band_pct!r}")
    band_low = int(round(TARGET_TRACK_PARAMS * (1.0 - PARAM_BAND_PCT / 100.0)))
    band_high = int(round(TARGET_TRACK_PARAMS * (1.0 + PARAM_BAND_PCT / 100.0)))
    if (band_low, band_high) != (BAND_LOW, BAND_HIGH):
        _err(
            "V543_PARAM_BUDGET",
            f"the ±5% band around 36,000 must be {BAND_LOW}–{BAND_HIGH}, "
            f"computed {band_low}–{band_high}",
        )
    for where, block in (("tracks.move", move), ("tracks.process", process)):
        if block.get("max_trainable_params") != PER_DECODER_CAP:
            _err(
                "V543_PARAM_BUDGET",
                f"{where}.max_trainable_params must be the 40,000 single-"
                f"candidate hard cap, got {block.get('max_trainable_params')!r}",
            )
    for key in ("move_refiner_params_max", "process_refiner_params_max"):
        _ival(resources, key, PER_DECODER_CAP, "V543_PARAM_BUDGET",
              "resources")
    _ival(resources, "total_model_params_max", TOTAL_MODEL_HARD_MAX,
          "V543_PARAM_BUDGET", "resources")
    total_target = ENCODER_FROZEN_PARAMS + TARGET_TRACK_PARAMS * 2
    total_band_max = ENCODER_FROZEN_PARAMS + BAND_HIGH * 2
    if total_target != TOTAL_MODEL_TARGET:
        _err("V543_PARAM_BUDGET",
             f"whole-model target must be 129,619, computed {total_target}")
    if total_band_max != TOTAL_MODEL_BAND_MAX:
        _err("V543_PARAM_BUDGET",
             f"in-band maximum must be 133,219, computed {total_band_max}")
    if TOTAL_MODEL_BAND_MAX >= TOTAL_MODEL_HARD_MAX:
        _err(
            "V543_PARAM_BUDGET",
            "the in-band maximum 133,219 must sit strictly below the 150,000 "
            "global hard cap",
        )
    # 57,619 = 55,392 active A0 backbone + 49 w_v + 2,178 process head; the
    # frozen component sum is asserted against the V542 migration invariant.
    if ENCODER_FROZEN_PARAMS != 55392 + 49 + 2178:
        _err(
            "V543_PARAM_BUDGET",
            "the frozen encoder component (57,619) must equal "
            "55,392 + 49 + 2,178 (V542 migration invariant)",
        )


def _check_training(cfg: Mapping[str, Any]) -> None:
    training = _block(cfg, "training", "V543_TRAINING")
    _str(training, "mode", "decoder_only", "V543_TRAINING", "training")
    _flag(training, "encoder_frozen", True, "V543_TRAINING", "training")
    _str(training, "optimizer", "AdamW", "V543_TRAINING", "training")
    _num(training, "learning_rate", 3.0e-4, "V543_TRAINING", "training")
    _ival(training, "warmup_updates", WARMUP_UPDATES, "V543_TRAINING",
          "training")
    _num(training, "weight_decay", 1.0e-4, "V543_TRAINING", "training")
    _num(training, "grad_clip", 1.0, "V543_TRAINING", "training")
    _str(training, "precision", "fp32", "V543_TRAINING", "training")
    _ival(training, "updates", DOSE_FLAT, "V543_TRAINING", "training")
    _ival(training, "val_every", 100, "V543_TRAINING", "training")
    _ival(training, "batch_curves", 8, "V543_TRAINING", "training")
    _ival(training, "frames_per_curve", 4, "V543_TRAINING", "training")
    _str(training, "sampler", SAMPLER_ID, "V543_TRAINING", "training")
    _str(training, "sampler_spec_sha256", SAMPLER_SPEC_SHA256, "V543_TRAINING",
         "training")
    _ival(training, "inner_val_max_samples", 8192, "V543_TRAINING", "training")
    _flag(training, "resume_requires_identity_match", True, "V543_TRAINING",
          "training")
    _flag(training, "torch_compile", False, "V543_TRAINING", "training")
    # dose identity: 37,900 flat + 400 warmup; 37,500 is derived, never a dose.
    warmup = training.get("warmup_updates")
    total = training.get("updates")
    if not isinstance(warmup, int) or not isinstance(total, int) \
            or isinstance(warmup, bool) or isinstance(total, bool):
        _err("V543_TRAINING",
             "warmup_updates/updates must be integers")
    if warmup != WARMUP_UPDATES or total != DOSE_FLAT:
        _err(
            "V543_TRAINING",
            f"dose must be {DOSE_FLAT} flat updates + {WARMUP_UPDATES} warmup, "
            f"got updates={total!r} warmup={warmup!r}",
        )
    if total - warmup != DERIVED_POST_WARMUP:
        _err(
            "V543_TRAINING",
            f"post-warmup dose must derive to {DERIVED_POST_WARMUP} "
            f"(37,900 - 400), got {total - warmup}",
        )
    if warmup >= total:
        _err("V543_TRAINING",
             "the 400-update warmup must be shorter than the 37,900 dose")


def _check_cache(cfg: Mapping[str, Any]) -> None:
    cache = _block(cfg, "cache", "V543_CACHE")
    _flag(cache, "enabled", True, "V543_CACHE", "cache")
    _str(cache, "mode", "frozen_encoder_only", "V543_CACHE", "cache")
    _ival(cache, "ram_lru_packets", 200000, "V543_CACHE", "cache")
    _ival(cache, "disk_max_gb", 30, "V543_CACHE", "cache")
    _ival(cache, "shard_curves", 256, "V543_CACHE", "cache")
    _num(cache, "hit_rate_target", 0.80, "V543_CACHE", "cache")
    _flag(cache, "joint_finetune_forbidden", True, "V543_CACHE", "cache")


def _check_screening(cfg: Mapping[str, Any]) -> None:
    screening = _block(cfg, "screening", "V543_SCREENING")
    _ival(screening, "smoke_updates", 200, "V543_SCREENING", "screening")
    _ival(screening, "diagnostic_updates", 2000, "V543_SCREENING", "screening")
    _seq(screening, "initial_formal_seeds", [T3_SEED, T4_SEED], "V543_SCREENING",
         "screening")
    _ival(screening, "finalist_seed", T5_SEED, "V543_SCREENING", "screening")
    _ival(screening, "finalists_per_track_max", 2, "V543_SCREENING",
          "screening")
    _flag(screening, "diagnostic_h3_selection_forbidden", True, "V543_SCREENING",
          "screening")
    _num(screening, "diagnostic_loss_drop_min_pct", 5.0, "V543_SCREENING",
         "screening")
    _num(screening, "diagnostic_move_dead_margin", 0.20, "V543_SCREENING",
         "screening")
    _num(screening, "diagnostic_move_rmse_rel_max_pct", 10.0, "V543_SCREENING",
         "screening")
    _num(screening, "diagnostic_process_auprc_drop_max", 0.02, "V543_SCREENING",
         "screening")


def _check_evaluation(cfg: Mapping[str, Any]) -> None:
    ev = _block(cfg, "evaluation", "V543_EVALUATION")
    _flag(ev, "matched_seed_baseline", True, "V543_EVALUATION", "evaluation")
    _str(ev, "baseline", "V543_M0_P0_recomputed", "V543_EVALUATION",
         "evaluation")
    _flag(ev, "cluster_bootstrap", True, "V543_EVALUATION", "evaluation")
    _ival(ev, "bootstrap_draws", BOOTSTRAP_DRAWS, "V543_EVALUATION",
          "evaluation")
    _ival(ev, "bootstrap_seed_root", BOOTSTRAP_SEED_ROOT, "V543_EVALUATION",
          "evaluation")
    _str(ev, "multiple_testing_correction", "Holm", "V543_EVALUATION",
         "evaluation")
    _flag(ev, "freeze_thresholds_before_runs", True, "V543_EVALUATION",
          "evaluation")
    _str(ev, "dev_policy", "final_frozen_candidates_only_one_shot",
         "V543_EVALUATION", "evaluation")
    _str(ev, "confirm_policy", "sealed", "V543_EVALUATION", "evaluation")
    _str(ev, "new_metric", "fixed_coverage_direction_at_3", "V543_EVALUATION",
         "evaluation")
    _str(ev, "q_corr_role", "diagnostic_only", "V543_EVALUATION", "evaluation")
    _ival(ev, "q_min_points_for_corr", 3, "V543_EVALUATION", "evaluation")
    _num(ev, "q_frac_curves_ge3_min", 0.70, "V543_EVALUATION", "evaluation")


def _check_thresholds(cfg: Mapping[str, Any], *, formal: bool) -> List[str]:
    """Appendix-B signed thresholds: exact key set, exact values, frozen mirror."""
    block = _block(cfg, "thresholds", "V543_THRESHOLD")
    unsigned: List[str] = []
    for key, expected in THRESHOLDS.items():
        if key not in block or block.get(key) is None:
            unsigned.append(key)
            continue
        value = block.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            _err("V543_THRESHOLD",
                 f"thresholds.{key} must be a finite number, got {value!r}")
        if float(value) != float(expected):
            _err(
                "V543_THRESHOLD",
                f"thresholds.{key} must stay at the signed value {expected} — "
                f"post-hoc threshold changes and token-preserving near-misses "
                f"are typed rejections, got {value!r}",
            )
    extra = sorted(set(block) - set(THRESHOLDS) - {"signoff"})
    if extra:
        _err(
            "V543_THRESHOLD",
            f"thresholds block carries key(s) {extra} outside the signed "
            "appendix-B table (relabelled near-miss keys are rejections)",
        )
    if unsigned:
        if formal:
            _err(
                "V543_THRESHOLD",
                f"formal gate: unsigned threshold(s) {unsigned} block the "
                "launch (thresholds are signed frozen 2026-11-02)",
            )
        return unsigned

    signoff = block.get("signoff")
    if not isinstance(signoff, Mapping):
        _err("V543_THRESHOLD", "thresholds.signoff must be a mapping")
    _str(signoff, "status", "frozen", "V543_THRESHOLD", "thresholds.signoff")
    _str(signoff, "signed_on", "2026-11-02", "V543_THRESHOLD",
         "thresholds.signoff")
    mirrored = {k: signoff.get(k) for k in THRESHOLDS
                if k in signoff}
    missing_mirror = sorted(set(THRESHOLDS) - set(mirrored))
    if missing_mirror:
        _err(
            "V543_THRESHOLD",
            f"thresholds.signoff must mirror all {len(THRESHOLDS)} signed "
            f"values; missing {missing_mirror}",
        )
    config_values = {k: block.get(k) for k in THRESHOLDS}
    if _canonical_json_bytes(mirrored) != _canonical_json_bytes(config_values):
        _err(
            "V543_THRESHOLD",
            "thresholds.signoff mirror is not byte-equal to the signed "
            "threshold values (int/float encoding drift included)",
        )
    # appendix B also signs the bootstrap statistics rows
    ev = _block(cfg, "evaluation", "V543_THRESHOLD")
    if ev.get("bootstrap_draws") != BOOTSTRAP_DRAWS:
        _err("V543_THRESHOLD",
             f"bootstrap_draws must be {BOOTSTRAP_DRAWS} (appendix B)")
    if ev.get("bootstrap_seed_root") != BOOTSTRAP_SEED_ROOT:
        _err("V543_THRESHOLD",
             f"bootstrap_seed_root must be {BOOTSTRAP_SEED_ROOT} (appendix B)")
    return []


def _check_resources(cfg: Mapping[str, Any]) -> None:
    res = _block(cfg, "resources", "V543_RESOURCE")
    _ival(res, "total_model_params_max", TOTAL_MODEL_HARD_MAX, "V543_RESOURCE",
          "resources")
    _ival(res, "move_refiner_params_max", PER_DECODER_CAP, "V543_RESOURCE",
          "resources")
    _ival(res, "process_refiner_params_max", PER_DECODER_CAP, "V543_RESOURCE",
          "resources")
    _num(res, "p95_latency_ratio_max", 2.0, "V543_RESOURCE", "resources")
    _ival(res, "encoder_forward_per_frame", 1, "V543_RESOURCE", "resources")
    _flag(res, "single_lane", True, "V543_RESOURCE", "resources")
    _flag(res, "swap_allowed", False, "V543_RESOURCE", "resources")
    _ival(res, "host_limit_gib", 12, "V543_RESOURCE", "resources")
    ev = _block(cfg, "evaluation", "V543_RESOURCE")
    if float(ev.get("bootstrap_draws") or 0) != float(BOOTSTRAP_DRAWS):
        _err("V543_RESOURCE", "bootstrap draws must stay 10,000")
    thr = _block(cfg, "thresholds", "V543_RESOURCE")
    if float(thr.get("max_latency_ratio_p95") or 0) != float(
            res.get("p95_latency_ratio_max")):
        _err(
            "V543_RESOURCE",
            "thresholds.max_latency_ratio_p95 must mirror "
            "resources.p95_latency_ratio_max (both 2.0)",
        )
    if float(thr.get("max_core_time_ratio") or 0) != 2.0:
        _err("V543_RESOURCE", "thresholds.max_core_time_ratio must stay 2.0")


def _check_conditional_generation(cfg: Mapping[str, Any]) -> None:
    cg = _block(cfg, "conditional_generation", "V543_CONDITIONAL")
    _seq(cg, "flow_nfe", [1, 2, 4, 8], "V543_CONDITIONAL",
         "conditional_generation")
    _str(cg, "flow_activation", "residual_bottleneck_evidence", "V543_CONDITIONAL",
         "conditional_generation")
    _flag(cg, "diffusion_default_enabled", False, "V543_CONDITIONAL",
          "conditional_generation")
    _str(cg, "graph_gru_activation", "process_systematic_failure",
         "V543_CONDITIONAL", "conditional_generation")
    _flag(cg, "sample_selection_uses_teacher", False, "V543_CONDITIONAL",
          "conditional_generation")
    _flag(cg, "activation_requires_new_signoff", True, "V543_CONDITIONAL",
          "conditional_generation")


def _check_run_matrix(cfg: Mapping[str, Any]) -> None:
    runs = _block(cfg, "run_matrix", "V543_RUN_MATRIX")
    _ival(runs, "formal_runs_max", FORMAL_RUNS_MAX, "V543_RUN_MATRIX",
          "run_matrix")
    _seq(runs, "t3_seed17", list(T3_RUN_DECODERS), "V543_RUN_MATRIX",
         "run_matrix")
    _seq(runs, "t4_seed29", list(T4_RUN_DECODERS), "V543_RUN_MATRIX",
         "run_matrix")
    _ival(runs, "t5_seed43_max", T5_MAX_RUNS, "V543_RUN_MATRIX", "run_matrix")
    _str(runs, "t6_selection", "two_track_champions_or_retain",
         "V543_RUN_MATRIX", "run_matrix")
    _seq(runs, "t7_integration", ["B0", "BM", "BP", "BMP"], "V543_RUN_MATRIX",
         "run_matrix")
    _str(runs, "t8_conditional", "separate_identity", "V543_RUN_MATRIX",
         "run_matrix")
    # seed assignment: T3 -> 17, T4 -> 29, T5 -> 43; two seeds screen, seed 43
    # only for the per-track finalists (plan §8.2, never seed shopping).
    t3 = tuple(runs.get("t3_seed17") or ())
    t4 = tuple(runs.get("t4_seed29") or ())
    if len(t3) != len(FORMAL_DECODER_IDS) or len(t4) != len(FORMAL_DECODER_IDS):
        _err(
            "V543_RUN_MATRIX",
            f"T3/T4 must each carry the {len(FORMAL_DECODER_IDS)} formal "
            f"decoders, got T3={list(t3)} T4={list(t4)}",
        )
    if set(t3) != set(t4):
        _err(
            "V543_RUN_MATRIX",
            "T3 (seed 17) and T4 (seed 29) must screen the same decoder set; "
            f"got T3={sorted(t3)} T4={sorted(t4)}",
        )
    unknown = sorted(set(t3 + t4) - set(FORMAL_DECODER_IDS))
    if unknown:
        _err(
            "V543_RUN_MATRIX",
            f"T3/T4 decoder id(s) {unknown} outside the frozen formal set "
            f"{list(FORMAL_DECODER_IDS)}",
        )
    scheduled = len(t3) + len(t4) + T5_MAX_RUNS
    if scheduled != FORMAL_RUNS_MAX:
        _err(
            "V543_RUN_MATRIX",
            f"7 + 7 + <=4 must equal the {FORMAL_RUNS_MAX}-run ceiling, "
            f"got {scheduled}",
        )
    # the exact appendix-C run-id set (18 rows, placeholders included)
    expected = EXPECTED_RUN_IDS
    derived = {
        *(f"V543_{d}_S{T3_SEED}" for d in t3),
        *(f"V543_{d}_S{T4_SEED}" for d in t4),
        "V543_<MOVE_top1>_S43",
        "V543_<MOVE_top2>_S43",
        "V543_<PROC_top1>_S43",
        "V543_<PROC_top2>_S43",
    }
    if derived != expected:
        _err(
            "V543_RUN_MATRIX",
            f"run-id set must be the exact appendix-C matrix; "
            f"missing={sorted(expected - derived)} "
            f"extra={sorted(derived - expected)}",
        )
    if len(derived) != EXPECTED_RUN_COUNT:
        _err(
            "V543_RUN_MATRIX",
            f"expected {EXPECTED_RUN_COUNT} distinct formal run ids, "
            f"got {len(derived)}",
        )
    dose = _block(cfg, "training", "V543_RUN_MATRIX").get("updates")
    total_updates = EXPECTED_RUN_COUNT * DOSE_FLAT
    if total_updates != TOTAL_FORMAL_UPDATES or total_updates != 682200:
        _err("V543_RUN_MATRIX",
             f"18 × 37,900 must equal 682,200, computed {total_updates}")
    if dose != DOSE_FLAT:
        _err("V543_RUN_MATRIX",
             f"every formal run trains {DOSE_FLAT} updates (training.updates "
             f"drift: {dose!r})")
    # M0/P0 are baselines: they never appear in the formal matrix.
    for baseline in ("M0", "P0", "P4-C", "M4", "M5", "P5"):
        if baseline in derived or any(baseline in rid for rid in derived):
            _err(
                "V543_RUN_MATRIX",
                f"{baseline} must never occupy a formal 18-run row "
                "(baseline / matched control / conditional identity)",
            )
    screening = _block(cfg, "screening", "V543_RUN_MATRIX")
    if tuple(screening.get("initial_formal_seeds") or ()) != (T3_SEED, T4_SEED):
        _err("V543_RUN_MATRIX",
             "screening.initial_formal_seeds must be [17, 29] (T3/T4)")
    if screening.get("finalist_seed") != T5_SEED:
        _err("V543_RUN_MATRIX",
             "screening.finalist_seed must be 43 (T5 finalists only)")
    if screening.get("finalists_per_track_max") != 2:
        _err("V543_RUN_MATRIX",
             "screening.finalists_per_track_max must be 2 (T5 <= 4 runs)")
    if FORMAL_STAGE_ORDER[0] != "T0_M0_P0_parity_three_seeds":
        _err("V543_RUN_MATRIX", "frozen stage order must start at T0 parity")


def _check_budget_table(cfg: Mapping[str, Any], *, formal: bool) -> List[str]:
    """``budget.per_candidate_core_time`` prereg-vs-formal semantics.

    The committed WP0 config keeps the table null (pre-WP7-profile): that
    state is a ``t1_blocked`` reason at prereg and a typed
    ``V543_BUDGET_NULL`` rejection at ``--formal``.  Once filled, the table
    must be a mapping covering all seven formal decoders with a finite
    POSITIVE core time; malformed tables (list/string, partial coverage,
    unknown decoder keys, null/zero/negative/NaN/bool entries) are typed
    ``V543_BUDGET_INVALID`` rejections.  Numeric value checks run at
    ``--formal`` only — the runner must never fill defaults.
    """
    budget = cfg.get("budget")
    if not isinstance(budget, Mapping):
        return []
    table = budget.get("per_candidate_core_time")
    if table is None or table == {} or table == []:
        if formal:
            _err(
                "V543_BUDGET_NULL",
                "formal gate: budget.per_candidate_core_time is still null — "
                "the same-env WP7 profile must be measured and the budget "
                "signed before any formal run (prereg mode reports this as "
                "t1_blocked; the runner must never fill defaults)",
            )
        # TODO(WP7): after the 200-update smoke / 2,000-update diagnostic /
        # 40+40 profile the table is filled per decoder from the same-env
        # M0/P0 denominator and re-hashed into budget_policy_sha256.
        return [
            "budget.per_candidate_core_time null: same-env WP7 profile signoff "
            "(M0/P0 denominator + 7 formal decoders) required before formal "
            "launch"
        ]
    if not isinstance(table, Mapping):
        _err(
            "V543_BUDGET_INVALID",
            "budget.per_candidate_core_time must be a mapping covering the "
            f"formal decoders {list(FORMAL_DECODER_IDS)} (a bare "
            f"{type(table).__name__} is not a signed budget table)",
        )
    extra = sorted(set(table) - KNOWN_DECODERS)
    if extra:
        _err(
            "V543_BUDGET_INVALID",
            f"budget.per_candidate_core_time carries unknown decoder(s) "
            f"{extra}; the frozen registry is {sorted(KNOWN_DECODERS)}",
        )
    missing_budget = sorted(set(FORMAL_DECODER_IDS) - set(table))
    if missing_budget:
        _err(
            "V543_BUDGET_INVALID",
            "budget.per_candidate_core_time must cover every formal decoder "
            f"(M0/P0 are the same-env cost denominator), missing "
            f"{missing_budget}",
        )
    if formal:
        for did in sorted(table):
            entry = table[did]
            value = entry.get("core_seconds") if isinstance(
                entry, Mapping) else entry
            if isinstance(value, bool) or not isinstance(
                    value, (int, float)) or not math.isfinite(value) \
                    or value <= 0:
                _err(
                    "V543_BUDGET_INVALID",
                    f"budget.per_candidate_core_time.{did} must carry a finite "
                    "POSITIVE core time (a number, or a mapping with a finite "
                    f"positive core_seconds); got {entry!r} — zero, negative, "
                    "NaN, inf, bool and null values are typed rejections",
                )
    single_lane = budget.get("single_lane")
    if single_lane is not True:
        _err("V543_RESOURCE",
             "budget.single_lane must be true (single-lane execution)")
    for key in ("swap_max", "oom_kill_max"):
        if budget.get(key) != 0:
            _err("V543_RESOURCE",
                 f"budget.{key} must be 0, got {budget.get(key)!r}")
    if budget.get("host_limit_gib") != 12:
        _err("V543_RESOURCE",
             f"budget.host_limit_gib must be 12, got "
             f"{budget.get('host_limit_gib')!r}")
    return []


def _check_governance_records(
    cfg: Mapping[str, Any], *, formal: bool, record_root: Optional[Path] = None
) -> List[str]:
    """Machine-check the preregistration signoff record instance.

    prereg: a missing/None instance is a ``t1_blocked`` reason (WP0 keeps the
    gate green while the signed instance lands); ``--formal`` typed-rejects.
    Content that does not mirror the frozen config is
    ``V543_RECORD_MISMATCH``; a missing/mismatched ``record_sha256`` is
    ``V543_RECORD_MISSING`` / ``V543_RECORD_MISMATCH``.
    """
    root = Path(record_root) if record_root is not None else PROJECT_ROOT
    spec = GOVERNANCE_RECORDS["preregistration_signoff_record"]
    path = root / spec["instance"]
    blocked: List[str] = []
    payload: Optional[Mapping[str, Any]] = None
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() \
            else None
    except (OSError, ValueError) as exc:
        raw = None
        if formal:
            _err("V543_RECORD_MISSING",
                 f"preregistration signoff instance unreadable: {path}: {exc}")
    if raw is None:
        reason = (
            f"preregistration_signoff_record instance absent at "
            f"{spec['instance']} (template is NOT sufficient; machine-checked "
            "at --formal)"
        )
        if formal:
            _err("V543_RECORD_MISSING", reason)
        return [reason]
    if not isinstance(raw, Mapping):
        if formal:
            _err("V543_RECORD_MISSING",
                 f"{spec['instance']} is not a JSON object")
        return [f"preregistration_signoff_record at {spec['instance']} is not "
                "a JSON object"]
    payload = raw
    missing = [f for f in spec["required_fields"] if payload.get(f) is None]
    if missing:
        if formal:
            _err(
                "V543_RECORD_MISSING",
                f"preregistration signoff instance lacks required field(s) "
                f"{missing} (a missing {RECORD_SHA_FIELD} is a "
                "V543_RECORD_MISSING rejection)",
            )
        return [f"preregistration_signoff_record missing field(s) {missing}"]

    expected = spec["expected"]
    for key, value in expected.items():
        if payload.get(key) != value:
            _err(
                "V543_RECORD_MISMATCH",
                f"preregistration signoff {key} must mirror the frozen config "
                f"({value!r}), got {payload.get(key)!r}",
            )
    roles = payload.get("signed_by_roles")
    if not isinstance(roles, Mapping) or not roles:
        _err("V543_RECORD_MISMATCH",
             "preregistration signoff signed_by_roles must be a non-empty "
             "mapping of the frozen role-key set")
    if set(roles) != SIGNOFF_ROLE_KEYS:
        _err(
            "V543_RECORD_MISMATCH",
            f"preregistration signoff signed_by_roles keys must be exactly "
            f"{sorted(SIGNOFF_ROLE_KEYS)}; missing="
            f"{sorted(SIGNOFF_ROLE_KEYS - set(roles))} "
            f"extra={sorted(set(roles) - SIGNOFF_ROLE_KEYS)}",
        )
    for role, signer in SIGNOFF_REQUIRED_ROLES.items():
        if roles.get(role) != signer:
            _err(
                "V543_RECORD_MISMATCH",
                f"preregistration signoff signed_by_roles.{role} must be "
                f"{signer!r}, got {roles.get(role)!r}",
            )
    for role in SIGNATORY_ROLES_NO_ORACLE:
        if roles.get(role) == SIGNOFF_REQUIRED_ROLES["oracle_reviewer"]:
            _err(
                "V543_RECORD_MISMATCH",
                f"the Oracle reviewer is NOT a signatory; signed_by_roles."
                f"{role} must not name "
                f"{SIGNOFF_REQUIRED_ROLES['oracle_reviewer']!r}",
            )
    signed = payload.get("thresholds_signed_frozen")
    if not isinstance(signed, Mapping):
        _err("V543_RECORD_MISMATCH",
             "preregistration signoff thresholds_signed_frozen must be a "
             "mapping mirroring the signed thresholds")
    config_thresholds = {
        k: (cfg.get("thresholds") or {}).get(k) for k in THRESHOLDS
    }
    mirrored = {k: signed.get(k) for k in THRESHOLDS}
    if _canonical_json_bytes(mirrored) != _canonical_json_bytes(
            config_thresholds):
        _err(
            "V543_RECORD_MISMATCH",
            "preregistration signoff thresholds_signed_frozen is not "
            "byte-equal to the config thresholds (all 19 signed keys; "
            "int/float encoding drift is a rejection)",
        )
    declared = payload.get(RECORD_SHA_FIELD)
    if not _sha256(declared):
        _err("V543_RECORD_MISSING",
             f"preregistration signoff must declare a 64-hex "
             f"{RECORD_SHA_FIELD}")
    actual = _record_canonical_sha256(payload)
    if declared != actual:
        _err(
            "V543_RECORD_MISMATCH",
            f"preregistration signoff {RECORD_SHA_FIELD} does not match the "
            f"canonical digest (declared {declared[:12]}..., "
            f"actual {actual[:12]}...) — stale digest or drifted payload",
        )
    return blocked


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def validate_config(
    cfg: Mapping[str, Any],
    *,
    formal: bool = False,
    check_gv: bool = True,
    record_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run every check; return a machine-readable report.

    Raises :class:`V543SchemaError` on the first typed rejection.
    ``record_root`` is a test seam for the governance-record check (defaults
    to PROJECT_ROOT when None).
    """
    checks: List[str] = []

    _check_schema(cfg)
    checks.append("schema")
    _check_campaign_status(cfg)
    checks.append("campaign_status")
    _check_required_fields(cfg)
    checks.append("required_fields")
    if check_gv:
        _check_gv_block(cfg)
        checks.append("gv_registry")
    _check_governance(cfg)
    checks.append("governance")
    _check_authorization(cfg)
    checks.append("authorization")
    _check_records_spec(cfg)
    checks.append("records_spec")
    _check_source(cfg)
    checks.append("source")
    _check_data(cfg)
    checks.append("data")
    _check_bridge(cfg)
    checks.append("bridge")
    _check_tracks(cfg)
    checks.append("tracks")
    _check_param_budget(cfg)
    checks.append("param_budget")
    _check_training(cfg)
    checks.append("training")
    _check_cache(cfg)
    checks.append("cache")
    _check_screening(cfg)
    checks.append("screening")
    _check_evaluation(cfg)
    checks.append("evaluation")
    null_thresholds = _check_thresholds(cfg, formal=formal)
    checks.append("thresholds")
    _check_resources(cfg)
    checks.append("resources")
    _check_conditional_generation(cfg)
    checks.append("conditional_generation")
    _check_run_matrix(cfg)
    checks.append("run_matrix")

    t1_blocked: List[str] = []
    if null_thresholds:
        t1_blocked.append(
            f"unsigned threshold(s) {null_thresholds} block the launch"
        )
    t1_blocked.extend(_check_budget_table(cfg, formal=formal))
    t1_blocked.extend(
        _check_governance_records(cfg, formal=formal, record_root=record_root)
    )
    checks.append("governance_records")

    return {
        "ok": True,
        "schema": SCHEMA_ID,
        "campaign_id": cfg.get("campaign_id"),
        "status": cfg.get("status"),
        "mode": "formal" if formal else "prereg",
        "checks": checks,
        "null_thresholds": null_thresholds,
        "t1_blocked": t1_blocked,
        "t1_config_ready": not t1_blocked,
    }


def load_and_validate(
    path: str | Path | None = None, *, formal: bool = False
) -> Dict[str, Any]:
    return validate_config(load_config(path), formal=formal)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--formal",
        action="store_true",
        help="formal gate: require non-null signed thresholds AND a signed "
        "budget table (null => V543_BUDGET_NULL; malformed table => "
        "V543_BUDGET_INVALID) AND the machine-checked governance record "
        "instance mirrored + record_sha256 verified (V543_RECORD_MISSING / "
        "V543_RECORD_MISMATCH typed rejections)",
    )
    parser.add_argument(
        "--prereg",
        action="store_true",
        help="explicit prereg mode (the default; the plan's WP0 CLI form)",
    )
    parser.add_argument("--json", action="store_true", help="print report as JSON")
    parser.add_argument(
        "--no-gv", action="store_true", help="skip the G–V registry cross-check"
    )
    args = parser.parse_args(argv)
    if args.formal and args.prereg:
        print("--prereg and --formal are mutually exclusive", file=sys.stderr)
        return 2
    try:
        report = validate_config(
            load_config(args.config), formal=args.formal, check_gv=not args.no_gv
        )
    except V543SchemaError as exc:
        if args.json:
            print(json.dumps({"ok": False, "code": exc.code, "detail": exc.detail}))
        else:
            print(f"V543 config rejected: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            "V543 config OK: schema={schema} status={status} mode={mode} "
            "checks={n}".format(n=len(report["checks"]), **report)
        )
        if report["t1_blocked"]:
            print("Formal launch not config-ready:")
            for reason in report["t1_blocked"]:
                print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
