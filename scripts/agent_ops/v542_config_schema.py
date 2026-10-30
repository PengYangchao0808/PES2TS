#!/usr/bin/env python3
"""V5.4.2 campaign config schema validator (WP0, ADR-0033 v1.2).

Validates ``config/campaigns/pes2ts_v542_backbone_50k.yaml`` (schema
``pes2ts_v542_campaign_v1``) and refuses to let a runner start from an
invalid config.  Independent from the V5.3-MP / V5.4.1 validators — this
module only reads the V542 yaml plus the frozen G–V registry and the
forward-view whitelist contract.

Frozen v1.1 contract highlights (Oracle gate-0 remediation, 2026-10-30),
extended by fix-5 (C1–C5/C7), fix-6 (C6 machine checks, 2026-10-30) and
the gate-2 Ruling 2 W_g/GRU-bias recording (2026-10-30):

* **exact 26-run matrix** — A0×3 matched baseline, A1–A4 seed17×4, A5
  seed17 conditional full-dose, top-2 preregistration A1–A4 × {29,43} = 8
  rows (max 4 activated), A5 supplement S29/S43 = 2 rows (max 2,
  parent gate ``a5_full_dose_complete_and_seed17_screen_pass``), 6
  unconditional diagnostic rows, A1-C/A5-C attribution rows; any
  missing/extra/renamed run is a typed rejection.
* **top-2 belongs to A1–A4 only** — the superseded A1–A5 × {29,43} = 10 /
  ``full_activation_count`` contract is rejected.
* **seed17 screen is not a three-seed H3 claim** — the frozen
  ``seed17_screen_pass`` predicate carries its own fields.
* **H3 statistics frozen** — ``bootstrap_seed_root=20261029``, deterministic
  sub-seed formula, 10,000 final draws, one-sided improvement p-value
  ``p = (1 + #{delta <= 0}) / 10001`` on the same paired draws as the CI,
  explicit Holm families, frame-level bootstrap forbidden.
* **R3-only warm start** — exact frozen path + ``R3_SHA256`` constant (no
  substring checks), sha256 seed recipe, construction order, per-candidate
  mapping tables, A1-C/A5-C ``init_source_id`` overrides.
* **identity/start** — ``source_manifest_sha256`` required, ``code_commit``
  optional, canonical ``t1_start_requires`` gate IDs + ``runplan_pin_record``,
  A0 reuse decision INSTANCE required before T1.
* **A0 sanity band is a hard T1/top2 precondition (fix-6 C5/C6)** —
  ``V542-A0-sanity-band`` sits in ``t1_start_requires`` AND in the top2
  freeze; ``run_matrix.formal_stage_order`` pins T0 A0×3 before any T1
  formal; a breach strictly blocks T1 (no exception signing).
* **final_selection_rule frozen (fix-6 C1)** — H3-qualified only, A0 and
  diagnostic variants excluded, 0 → A0 retained, 1 → that candidate, many →
  the exact 7-step lexicographic tuple, ``dev_used=false``, report all
  qualified, rule block inside the statistics SHA, A5-C uses the selected
  candidate only.
* **route_logic parenthesized (fix-6 C2)** — MOVE/PROCESS ``pass_logic``
  strings and component thresholds are exact; the seed17 screen reuses the
  same parenthesized logic (no dangling OR).
* **statistics fields (fix-6 C6; partially exact)** — ``ci=percentile_95``,
  the pairing sentence, ``final_draws_ref=thresholds.final_bootstrap_draws``,
  all four improvement directions, the p numerator, CI/p same draws,
  frame-level forbidden, and the Holm families/alpha are checked exactly.
  **Still token-level (gate-0 advisory A2 open):** ``h3_eligible_candidates``
  (H2 / diagnostic / 三 seed / winner-eligible token presence) and
  ``champion_rule`` (reference to ``evaluation_contract.final_selection_rule``);
  both are scheduled for exact/structured freezing before P3/WP6.
* **parity asset tiers (fix-6 C3; gate-0 A6)** — 6/6 rows carry
  ``source_tier`` + ``run_updates`` (source-run training dose: V541 E0 =
  37,900; V53MP C = 15,400) + ``checkpoint_update`` (the pinned payload's
  best-checkpoint ``update``: 37000/36400/37800; 15400/15300/15400) with
  ``checkpoint_update <= run_updates``; ``training_updates`` is kept ONLY
  as an alias equal to ``run_updates`` (single run-dose meaning — never
  the checkpoint update); smoke and non-pinned checkpoints forbidden;
  GPU 1e-6/1e-5 hard caps, floor>cap ⇒ inconclusive, no relaxation.
* **W_g / GRUCell bias recorded per candidate (gate-2 Ruling 2;
  2026-10-30)** — ``w_g_bias`` must be exactly ``true`` for A2/A4 (the
  biased ``nn.Linear(5, 48)`` graph-conditioning injection = 288 params)
  and ``false`` for A5/A5-C (``bias=False`` = 240 params per cell);
  ``gru_cell_bias`` records the per-cell GRUCell bias in cell order — A5
  ``[true]`` (the single biased ``GRUCell(48, 48)`` = 14,112) and A5-C
  ``[true, false]`` (cell 0 = the A5 recipe; cell 1
  ``GRUCell(48, 48, bias=False)`` = 13,824).  These choices reproduce the
  frozen preregistered totals EXACTLY (A2 54,896 / A4 55,840 / A5 31,584
  / A5-C 53,712 — unchanged invariants); ``W_g`` is zero-initialised so
  the variants are identical at warm start.  Missing fields ⇒
  ``V542_MISSING_FIELD``; value drift (including token-preserving
  string/int near-misses such as ``"true"``/``1``) ⇒
  ``V542_PARAM_BUDGET``; candidates WITHOUT a ``W_g`` injection or GRU
  cells (A0/A1/A1-C/A3) must not carry the fields at all.

Typed rejections (each carries a stable ``code``; 19 codes — the
authoritative count for docs/registry wording):

==========================  ==========================================================
code                          rejection
==========================  ==========================================================
``V542_CONFIG_MISSING``       file absent / unreadable / not a mapping
``V542_SCHEMA``               top-level ``schema`` != ``pes2ts_v542_campaign_v1``
``V542_MISSING_FIELD``        a required field is absent or None
``V542_OBJECTIVE``            ``objective`` != ``unified_move_process``
``V542_ARM_MATRIX``           legacy multi-arm matrix exposed (``arm``/``arms`` keys)
``V542_BACKBONE_UNKNOWN``     backbone id outside the frozen A0–A5 (+A1-C/A5-C) registry
``V542_FORWARD_KEY``          forward-view key outside the contract whitelist
``V542_INTERFACE``            BackboneOutput/Decoder/PROCESS interface contract broken
``V542_PARAM_BUDGET``         parameter invariants broken (A0 55392/57619, 49, 2178, <=70k) or W_g/GRU-bias record drift (gate-2 Ruling 2)
``V542_INIT_SOURCE``          warm-start source not the frozen R3 checkpoint / policy drift
``V542_A0_REUSE``             a0_baseline.decision != retrain_a0_x3 (or shadow-run required)
``V542_CONFIRM``              confirm/sealed/QC authorization not false/0
``V542_RESOURCE``             single-lane / swap / OOM guards violated
``V542_THRESHOLD_NULL``       a signed threshold is null while ``--formal`` requested
``V542_BUDGET_NULL``          ``--formal`` with budget.per_candidate_core_time still null
``V542_GV_MISMATCH``          ``gv`` block disagrees with ``config/registry/g_v_map.yaml``
``V542_RUN_MATRIX``           run id / class / seed / activation outside preregistered shape
``V542_SAMPLER_DRIFT``        ``sampler_binding`` != the live ``v541_strata`` spec
``V542_R3_INIT``              R3 init path/SHA absent, malformed, drifted or unverifiable
==========================  ==========================================================

Modes
-----
* ``prereg`` (default) — structural validation; a null budget is reported as
  a ``t1_blocked`` reason (WP0 state: budget filled only after the same-env
  A0 profile; the runner never fills defaults).
* ``formal`` — additionally requires every signed threshold to be non-null
  AND the budget table to be signed (typed ``V542_BUDGET_NULL`` rejection
  while it is still null), i.e. the gate a formal launch must pass.

``validate_config`` runs these named checks in order (26 checks with the
G–V cross-check enabled — keep this list current):

``schema``, ``objective``, ``required_fields``, ``backbone_registry``,
``forward_view``, ``interface``, ``param_budget``, ``init_source``,
``a0_decision``, ``authorization``, ``sampler_binding``, ``dose_blocks``,
``selection_tau``, ``gv_registry``, ``identity_start``, ``run_matrix``,
``top2_preregistration``, ``activation_branches``, ``statistics``,
``route_logic``, ``final_selection``, ``seed17_screen``,
``a0_sanity_band``, ``run_outcomes``, ``cost_acceptance``, ``thresholds``.

CLI::

    python scripts/agent_ops/v542_config_schema.py                 # prereg mode
    python scripts/agent_ops/v542_config_schema.py --formal        # formal gate
    python scripts/agent_ops/v542_config_schema.py --config PATH --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # PyYAML: existing repo dependency

from pes2ts_core.gv_registry import (  # noqa: E402
    DEFAULT_MAP_PATH,
    GVMapError,
    load_gv_map,
    resolve_campaign,
)
from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    FORWARD_VIEW_KEYS_V3,
)

__all__ = [
    "A0_ACTIVE_PARAMS",
    "A0_BREACH_PROCEDURE",
    "A0_JOINT_PARAMS",
    "A0_SANITY_BAND_GATE",
    "A0_SANITY_BAND_NOTE_EXACT",
    "A0_SANITY_BAND_VALUES",
    "A5_SUPPLEMENT_MAX_RUNS",
    "A5_SUPPLEMENT_PARENT_GATE",
    "BACKBONE_ACTIVE_MAX",
    "BIAS_FIELD_CANDIDATES",
    "BOOTSTRAP_SEED_ROOT",
    "CAMPAIGN_ID",
    "CI_METHOD",
    "CONSTRUCTION_ORDER",
    "DECODER_PARAMS",
    "DEFAULT_CONFIG",
    "EXPECTED_RUN_IDS",
    "EXPECTED_RUN_TABLE",
    "FINAL_DRAWS",
    "FINAL_DRAWS_REF",
    "FINAL_EFFECT_MARGIN",
    "FINAL_SELECTION_EXCLUDED",
    "FINAL_SELECTION_NAME",
    "FINAL_SELECTION_TIE_BREAK",
    "FORMAL_STAGE_ORDER",
    "FORMAL_THRESHOLDS",
    "GPU_ATOL",
    "GPU_RTOL",
    "GRU_CELL_BIAS",
    "GV_CAMPAIGN_ID",
    "IDENTITY_REQUIRED_FIELDS",
    "IMPROVEMENT_DIRECTIONS",
    "INIT_MAPPING_AUDIT_EXACT",
    "INIT_SEED_RECIPE",
    "KNOWN_BACKBONES",
    "MOVE_PASS_LOGIC",
    "OBJECTIVE",
    "PAIRING_SENTENCE",
    "PARITY_ASSETS",
    "PARITY_ASSET_TIERS",
    "PARITY_TIER_POLICY_FORBIDDEN",
    "P_FORMULA_EXACT",
    "PROCESS_PARAMS",
    "PROCESS_PASS_LOGIC",
    "R3_REL_PATH",
    "R3_SHA256",
    "REQUIRED_FIELDS",
    "ROUTE_COMPONENTS",
    "SCHEMA_ID",
    "T1_CANONICAL_GATES",
    "T1_RECORD_WHITELIST",
    "T1_START_REQUIRES",
    "T3_MAX_RUNS",
    "TOP2_CANDIDATES",
    "TOP2_MAX_RUNS",
    "TYPED_REJECTIONS",
    "V542SchemaError",
    "W_G_BIAS",
    "load_config",
    "main",
    "validate_config",
]

SCHEMA_ID = "pes2ts_v542_campaign_v1"
CAMPAIGN_ID = "PES2TS_V542_BACKBONE_50K"
GV_CAMPAIGN_ID = "V5.4.2"
OBJECTIVE = "unified_move_process"

#: Frozen parameter invariants (ADR-0033 §2.1; Oracle §2.0/§2.6).
A0_ACTIVE_PARAMS = 55392
A0_JOINT_PARAMS = 57619
DECODER_PARAMS = 49
PROCESS_PARAMS = 2178
BACKBONE_ACTIVE_MAX = 70000
Z_DIM = 64
EDGE_MSG_DIM = 48

#: The 19 typed rejection codes above — the authoritative count for docs.
TYPED_REJECTIONS: Tuple[str, ...] = (
    "V542_CONFIG_MISSING",
    "V542_SCHEMA",
    "V542_MISSING_FIELD",
    "V542_OBJECTIVE",
    "V542_ARM_MATRIX",
    "V542_BACKBONE_UNKNOWN",
    "V542_FORWARD_KEY",
    "V542_INTERFACE",
    "V542_PARAM_BUDGET",
    "V542_INIT_SOURCE",
    "V542_A0_REUSE",
    "V542_CONFIRM",
    "V542_RESOURCE",
    "V542_THRESHOLD_NULL",
    "V542_BUDGET_NULL",
    "V542_GV_MISMATCH",
    "V542_RUN_MATRIX",
    "V542_SAMPLER_DRIFT",
    "V542_R3_INIT",
)

#: Frozen backbone registry: id -> (family, in_main_matrix, diagnostic_only,
#: estimated_params).  A0 numbers are migration invariants; A1–A5/A1-C/A5-C
#: numbers are design ESTIMATES — the validator enforces the <=70k cap and
#: the estimated flag, never exact equality for them.
BACKBONE_REGISTRY: Dict[str, Tuple[str, bool, bool, bool]] = {
    "A0": ("legacy_star_3l_adapter", True, False, False),
    "A1": ("geoegnn_lite_2l", True, False, True),
    "A2": ("painn_lite_single_frame_v32", True, False, True),
    "A3": ("egnn_reaction_query_2x24", True, False, True),
    "A4": ("sparse_geometry_transformer_4x12", True, False, True),
    "A5": ("graph_gru_shared_cell", True, False, True),
    "A1-C": ("geoegnn_lite_3l", False, True, True),
    "A5-C": ("graph_gru_2cell_alternating", False, True, True),
}
KNOWN_BACKBONES = frozenset(BACKBONE_REGISTRY)
#: Backbones that may occupy winner-eligible formal runs.
WINNER_BACKBONES = frozenset({"A1", "A2", "A3", "A4", "A5"})
#: Diagnostic-only backbones may never be winner-eligible nor in T1/T3.
DIAGNOSTIC_BACKBONES = frozenset({"A1-C", "A5-C"})

# ---------------------------------------------------------------------------
# gate-2 Ruling 2 (2026-10-30): per-candidate W_g / GRUCell bias freeze.
# A2/A4 inject graph conditioning through a BIASED nn.Linear(5, 48)
# (288 params); A5/A5-C use nn.Linear(5, 48, bias=False) (240 params per
# cell).  In the same exact-total arithmetic A5's single GRUCell(48, 48)
# is biased (14,112 params) and A5-C's SECOND GRUCell is bias-free
# (13,824; cell 0 keeps the A5 recipe).  These choices reproduce the
# frozen preregistered totals EXACTLY (A2 54,896 / A4 55,840 / A5 31,584
# / A5-C 53,712 — never padded/shrunk); W_g is zero-initialised, so at
# warm start both variants are identical and the difference is a small
# per-candidate capacity choice (48 params/cell), recorded here so it can
# never drift silently.  Candidates without a W_g injection (A0/A1/A1-C/
# A3) or without GRU cells must not carry the fields.
# ---------------------------------------------------------------------------
W_G_BIAS: Dict[str, bool] = {"A2": True, "A4": True, "A5": False, "A5-C": False}
GRU_CELL_BIAS: Dict[str, Tuple[bool, ...]] = {
    "A5": (True,),
    "A5-C": (True, False),
}
#: Every candidate that must record exactly one of the two bias fields.
BIAS_FIELD_CANDIDATES = frozenset(W_G_BIAS)

#: Signed thresholds that must be non-null before formal launch (28 keys:
#: the original 24 + bootstrap_seed_root/top2_max_runs/a5_supplement_max_runs/
#: t3_max_runs added by the v1.1 statistics/budget remediation).
FORMAL_THRESHOLDS: Tuple[str, ...] = (
    "sign_drop_abs_max",
    "dead_fraction_increase_abs_max",
    "fixed_coverage_direction_regression_rel_max_pct",
    "rmse_zero_regression_rel_max_pct",
    "other_endpoint_auprc_drop_abs_max",
    "other_q_corr_drop_abs_max",
    "h3_move_direction_rel_min_pct",
    "h3_move_direction_abs_min",
    "h3_move_rmse_rel_min_pct",
    "h3_process_delta_auprc_abs_min",
    "h3_process_delta_q_corr_abs_min",
    "move_exit_alpha_each",
    "diagnostic_loss_drop_min_pct",
    "diagnostic_dead_fraction_margin_vs_a0",
    "diagnostic_rmse_zero_rel_max_pct",
    "gpu_parity_atol",
    "gpu_parity_rtol",
    "max_core_time_ratio",
    "final_bootstrap_draws",
    "bootstrap_seed_root",
    "top2_max_runs",
    "a5_supplement_max_runs",
    "t3_max_runs",
    "backbone_active_max",
    "a0_backbone_active",
    "a0_joint_params",
    "move_decoder_params",
    "process_head_params",
)

REQUIRED_FIELDS: Tuple[str, ...] = (
    "schema",
    "campaign_id",
    "version",
    "objective",
    "gv",
    "governance",
    "data_contract",
    "split_contract",
    "identity_contract",
    "backbone_contract",
    "training_contract",
    "a0_baseline",
    "smoke",
    "profile",
    "diagnostic",
    "run_matrix",
    "evaluation_contract",
    "gates",
    "thresholds",
    "cost_acceptance",
    "budget",
    "run_outcomes",
    "plan_outputs",
)

ARM_MATRIX_KEYS = ("arm", "arms", "arm_matrix")

#: Frozen R3 warm-start identity (V5.4.1 config freeze; re-verified on load).
R3_RUN_ID = "5372d9b12807c3f0"
R3_REL_PATH = "outputs/pesn/30h_campaign/r3_runs/5372d9b12807c3f0/checkpoint_best.pt"
R3_SHA256 = "a2e46b6ad65cc18996d6c1660c5ae24c723397ca22526e52fa34e430afd0417f"

# ---------------------------------------------------------------------------
# v1.1 frozen selection contract (ADR-0033 v1.1 §2.6; Oracle findings 1–3)
# ---------------------------------------------------------------------------
#: top-2 preregistration belongs to A1–A4 ONLY — A5 never enters top2.
TOP2_CANDIDATES: Tuple[str, ...] = ("A1", "A2", "A3", "A4")
T3_SEEDS: Tuple[int, ...] = (29, 43)
TOP2_LISTED_RUNS = 8          # 2 candidates x 2 seeds
TOP2_MAX_RUNS = 4             # <=2 activated candidates x 2 seeds
A5_SUPPLEMENT_MAX_RUNS = 2    # A5 T3 supplement (separate branch)
T3_MAX_RUNS = 6               # top2(<=4) + a5_supplement(<=2)
T3_ACTIVATION = "activated_only_by_inner_val_top2"
A5_SUPPLEMENT_CONDITION = "a5_full_dose_complete_and_seed17_screen_pass"
A5_SUPPLEMENT_PARENT_GATE = "a5_full_dose_complete_and_seed17_screen_pass"
T3_UNACTIVATED_STATUS = "gate_skipped"
T3_RUN_ID_TOKEN = "TOP2"  # ambiguous placeholder ids are forbidden
SCREEN_PREDICATE_REF = "evaluation_contract.seed17_screen_pass_predicate"

A5_FULLDOSE_CONDITION = "no_A1_A4_h3_qualified_candidate_after_top2_seeds"
A5_FULLDOSE_LAUNCH_REQUIRES: Tuple[str, ...] = (
    "diagnostic_gate", "cost_gate", "H0_legality", "H1_resource",
)
#: Frozen spec override 2026-10-29: A1-C auto-activates when A1 enters the
#: top-2 supplement set OR is selected as Backbone* — NOT top-ranked only.
A1C_ACTIVATION = "a1_in_top2_supplement_set_OR_a1_selected_as_backbone"
A1C_SKIP_REASON = "a1_not_in_top2_and_not_selected"
#: v1.1 narrowing: A5-C activates ONLY when A5 is SELECTED as Backbone*.
A5C_ACTIVATION = "a5_selected_backbone"
A5C_SKIP_REASON = "a5_not_selected_backbone"
ATTR_ACTIVATION_REQUIRES: Tuple[str, ...] = (
    "H0_fixture", "smoke_200_conditional", "profile_40_40", "cost_gate",
    "budget_record",
)
A0_DECISION = "retrain_a0_x3"

RUN_CLASSES = frozenset({"nominal", "conditional", "diagnostic"})
OUTCOME_VOCABULARY = frozenset(
    {"done", "invalid_run", "scientific_failure", "resource_failure",
     "budget_censored", "gate_skipped"}
)


def _expected_run_table() -> Dict[str, Dict[str, Any]]:
    """The frozen exact 26-run matrix (ADR-0033 v1.1; plan §9).

    run_id -> exact required fields.  ``new_training`` marks unconditional
    new trainings; conditional rows instead carry ``new_training_if_activated``
    plus ``unactivated_status=gate_skipped``.
    """
    table: Dict[str, Dict[str, Any]] = {}
    for seed in (17, 29, 43):
        table[f"V542_A0_S{seed}"] = {
            "stage": "T0", "class": "nominal", "dose": "formal",
            "backbone": "A0", "seed": seed, "winner_eligible": False,
            "new_training": 1, "role": "matched_baseline_retrain",
        }
    for bid in TOP2_CANDIDATES:
        table[f"V542_{bid}_S17"] = {
            "stage": "T1", "class": "nominal", "dose": "formal",
            "backbone": bid, "seed": 17, "winner_eligible": True,
            "new_training": 1,
            "launch_requires": ["diagnostic_gate", "cost_gate"],
        }
    table["V542_A5_S17"] = {
        "stage": "T2", "class": "conditional", "dose": "formal",
        "backbone": "A5", "seed": 17, "winner_eligible": True,
        "new_training_if_activated": 1, "unactivated_status": "gate_skipped",
        "condition": A5_FULLDOSE_CONDITION,
        "launch_requires": list(A5_FULLDOSE_LAUNCH_REQUIRES),
    }
    for bid in TOP2_CANDIDATES:
        for seed in T3_SEEDS:
            table[f"V542_{bid}_S{seed}"] = {
                "stage": "T3", "class": "conditional", "dose": "formal",
                "backbone": bid, "seed": seed, "winner_eligible": True,
                "new_training_if_activated": 1,
                "unactivated_status": "gate_skipped",
                "condition": T3_ACTIVATION,
            }
    for seed in T3_SEEDS:
        table[f"V542_A5_S{seed}"] = {
            "stage": "T3", "class": "conditional", "dose": "formal",
            "backbone": "A5", "seed": seed, "winner_eligible": True,
            "new_training_if_activated": 1,
            "unactivated_status": "gate_skipped",
            "condition": A5_SUPPLEMENT_CONDITION,
            "parent_gate": A5_SUPPLEMENT_PARENT_GATE,
        }
    table["V542_DIAG_A0_S17"] = {
        "stage": "T-DIAG", "class": "diagnostic", "dose": "diagnostic_2000",
        "backbone": "A0", "seed": 17, "winner_eligible": False,
        "new_training": 1, "role": "matched_dose_reference",
    }
    for bid in ("A1", "A2", "A3", "A4", "A5"):
        table[f"V542_DIAG_{bid}_S17"] = {
            "stage": "T-DIAG", "class": "diagnostic", "dose": "diagnostic_2000",
            "backbone": bid, "seed": 17, "winner_eligible": False,
            "new_training": 1,
        }
    table["V542_A1C_S17"] = {
        "stage": "T-ATTR", "class": "diagnostic", "dose": "formal",
        "backbone": "A1-C", "seed": 17, "winner_eligible": False,
        "new_training_if_activated": 1, "unactivated_status": "gate_skipped",
        "condition": A1C_ACTIVATION,
        "activation_requires": list(ATTR_ACTIVATION_REQUIRES),
        "skip_reason": A1C_SKIP_REASON,
    }
    table["V542_A5C_S17"] = {
        "stage": "T-ATTR", "class": "diagnostic", "dose": "formal",
        "backbone": "A5-C", "seed": 17, "winner_eligible": False,
        "new_training_if_activated": 1, "unactivated_status": "gate_skipped",
        "condition": A5C_ACTIVATION,
        "activation_requires": list(ATTR_ACTIVATION_REQUIRES),
        "skip_reason": A5C_SKIP_REASON,
    }
    return table


#: Exact frozen 26-run matrix — run_id -> required field/value map.
EXPECTED_RUN_TABLE: Dict[str, Dict[str, Any]] = _expected_run_table()
EXPECTED_RUN_IDS = frozenset(EXPECTED_RUN_TABLE)
EXPECTED_RUN_COUNT = 26

#: Frozen parity assets (path + SHA-256 verified by the read-only lane
#: 2026-10-30; see .slim/deepwork/v542-backbone-20261029/parity-assets.md).
#: 6/6 required; smoke checkpoints are forbidden as parity assets.
PARITY_ASSETS: Tuple[Tuple[str, str, str], ...] = (
    ("V541_E0_S17",
     "outputs/pesn/v541_embed_50k/runs/V541_T0R_E0_S17/checkpoint_best.pt",
     "c1e35610d33e2350ec5987ccbced07ba7e665458e6078a7eaf2e0de9e90ca56d"),
    ("V541_E0_S29",
     "outputs/pesn/v541_embed_50k/runs/V541_T0R_E0_S29/checkpoint_best.pt",
     "a214d6e9c9cf4a80bbf079567d13c0a09061dcbd0b5b2f3dec4f53f9cc29e49d"),
    ("V541_E0_S43",
     "outputs/pesn/v541_embed_50k/runs/V541_T0R_E0_S43/checkpoint_best.pt",
     "92412a1af295cc58c880c32bf9ee0ea653b46ef31648fc9cf1cc76a806a6f00c"),
    ("V53MP_C_S17",
     "outputs/pesn/v53mp/v53mp_20k_C_s17/checkpoint_best.pt",
     "871596e61b7d2e2ed77328eaa884b848621f07533e8ffd242a98442c0aec5db3"),
    ("V53MP_C_S29",
     "outputs/pesn/v53mp/v53mp_20k_C_s29/checkpoint_best.pt",
     "d456050e9ed5f231691306097e120459cd8679768da30313c837ef1307f06884"),
    ("V53MP_C_S43",
     "outputs/pesn/v53mp/v53mp_20k_C_s43/checkpoint_best.pt",
     "d6b11a20c90cda353400639a2f88c54a4ba3b91fcda55a67241d8fa967f4715b"),
)
PARITY_ASSET_COUNT = 6
PARITY_SMOKE_PATH_TOKEN = "/smoke/"

# ---------------------------------------------------------------------------
# v1.1 frozen statistics contract (ADR-0033 v1.1 §2.5; Oracle finding 3)
# ---------------------------------------------------------------------------
BOOTSTRAP_SEED_ROOT = 20261029
BOOTSTRAP_SUBSEED_FORMULA = (
    'int(sha256(f"v542|{root}|{seed}|{endpoint}").hexdigest()[:16], 16) '
    "% 2147483647"
)
FINAL_DRAWS = 10000
INITIAL_DRAWS = 1000
P_VALUE_TYPE = "one_sided_improvement_paired_cluster_bootstrap"
#: fix-6 C6: exact structured statistics fields (Oracle finding 6).
CI_METHOD = "percentile_95"
PAIRING_SENTENCE = "same cluster same frame set; both models share resample indices"
FINAL_DRAWS_REF = "thresholds.final_bootstrap_draws"
P_FORMULA_EXACT = (
    "p = (1 + #{seed-averaged paired bootstrap improvement <= 0}) / (10001)"
)
IMPROVEMENT_DIRECTIONS: Dict[str, str] = {
    "direction": "candidate_minus_a0",
    "endpoint_auprc": "candidate_minus_a0",
    "q_corr": "candidate_minus_a0",
    "rmse_zero": "a0_minus_candidate",
}

# ---------------------------------------------------------------------------
# fix-6 C1: frozen final_selection_rule (ADR-0033 v1.2 §2.5; Oracle finding 1)
# ---------------------------------------------------------------------------
FINAL_SELECTION_NAME = "final_selection_rule"
FINAL_SELECTION_EXCLUDED: Tuple[str, ...] = (
    "A0_matched_baseline",
    "diagnostic_variants_A1-C_A5-C",
)
#: Exact lexicographic tie-break tuple — order is part of the freeze.
FINAL_SELECTION_TIE_BREAK: Tuple[str, ...] = (
    "1_route_priority_MOVE_qualified_before_PROCESS_only",
    "2_min_holm_adjusted_p_on_qualifying_confirmatory_endpoint",
    "3_max_effect_margin_on_qualifying_endpoint",
    "4_min_core_time_ratio",
    "5_min_active_backbone_params",
    "6_min_inner_val_total_loss",
    "7_min_candidate_id_lexicographic",
)
FINAL_EFFECT_MARGIN: Dict[str, str] = {
    "move_direction": "min(relative_effect/0.03, absolute_effect/0.01)",
    "move_rmse": "relative_improvement/0.03",
    "process_auprc": "delta/0.02",
    "process_q": "delta/0.03",
}

# ---------------------------------------------------------------------------
# fix-6 C2: frozen parenthesized route logic (ADR-0033 v1.2 §2.5; finding 2)
# ---------------------------------------------------------------------------
MOVE_PASS_LOGIC = (
    "(direction_pass AND rmse_non_inferior) OR "
    "(rmse_pass AND direction_non_inferior)"
)
PROCESS_PASS_LOGIC = (
    "(auprc_pass AND q_non_inferior) OR (q_pass AND auprc_non_inferior)"
)
#: component -> exact frozen definition text (thresholds are part of the freeze)
ROUTE_COMPONENTS: Dict[str, Dict[str, str]] = {
    "move_route": {
        "direction_pass": (
            "fixed_coverage_direction 相对改善 >=3% 且绝对改善 >=0.01"
        ),
        "rmse_pass": "rmse_zero 相对改善 >=3%",
        "rmse_non_inferior": "rmse_zero 退化 <=2%",
        "direction_non_inferior": "fixed_coverage_direction 退化 <=2%",
    },
    "process_route": {
        "auprc_pass": "endpoint_auprc delta >= 0.02",
        "q_pass": "q_corr delta >= 0.03",
        "auprc_non_inferior": "endpoint_auprc 下降 <=0.01",
        "q_non_inferior": "q_corr 下降 <=0.02",
    },
}

# ---------------------------------------------------------------------------
# fix-6 C5: A0 sanity band hard precondition (Oracle finding 5)
# ---------------------------------------------------------------------------
A0_SANITY_BAND_GATE = "V542-A0-sanity-band"
A0_SANITY_BAND_VALUES: Dict[str, Any] = {
    "rmse_zero_rel_max_pct": 2.0,
    "auprc_abs_max": 0.01,
    "q_corr_abs_max": 0.01,
    "on_breach": "block_T1_and_migration_audit",
}
#: Frozen formal stage order — T0 A0x3 + sanity band BEFORE any T1 formal.
FORMAL_STAGE_ORDER: Tuple[str, ...] = (
    "T0_V542_A0_S17_S29_S43_formal_plus_V542-A0-sanity-band",
    "T1_A1-A4_S17_formal",
    "T2_A5_S17_conditional_then_T3_top2_and_a5_supplement",
    "T-ATTR_A1C_A5C_conditional",
)
#: Breach procedure steps — step 3 names the identity layer per defect
#: class (gate-0 A8); step 5 forbids exception signing (no escape hatch).
A0_BREACH_PROCEDURE: Tuple[str, ...] = (
    "1_strictly_block_all_T1_formal_candidate_launches",
    "2_migration_audit: parity_certificate + process_head_anchor + r3_mapping "
    "+ sampler + evaluator + code + data + initialization",
    "3_if_code_data_or_init_defect_found: fix_defect_then_bump_matching_"
    "identity_layer (code_defect=>source_tree_fingerprint; "
    "data_defect=>data_manifest_sha256; init_defect=>init_mapping_sha256; "
    "anchor_implicated=>process_head_anchor_sha256) then "
    "rerun_A0_x3_formal; old_A0x3=>invalid_run, no exception",
    "4_resume_T1_only_after_V542-A0-sanity-band_pass_on_clean_A0_x3",
    "5_sanity_exception_signing_forbidden_under_breach",
)
#: gate-0 A8: the sanity-band note frozen EXACTLY — one rule, no
#: contradiction with the breach procedure: per-defect-class identity
#: bump (code => source_tree_fingerprint, data => data_manifest_sha256,
#: init => init_mapping_sha256, anchor implicated =>
#: process_head_anchor_sha256), rerun A0x3, old A0x3 => invalid_run, and
#: sanity-exception signing stays forbidden under breach.
A0_SANITY_BAND_NOTE_EXACT: str = (
    "带宽只抓迁移大 bug, 不要求与 E0 数值相同; breach = 严格阻断 T1 + 迁移审计 + "
    "按缺陷类别 bump 对应身份层 (代码缺陷=>source_tree_fingerprint; "
    "数据缺陷=>data_manifest_sha256; 初始化缺陷=>init_mapping_sha256; "
    "anchor 牵连=>process_head_anchor_sha256) 后重跑 A0×3 formal "
    "(旧 A0×3 记 invalid_run, 不混比); 禁止签 exception 继续"
)
#: gate-0 A8: init_policy.mapping_audit frozen EXACTLY — the declared
#: w_v dual-write alias (backbone.w_v <-> move_decoder.w_v, the same
#: tensor object) is the single legitimate non-identity write; a strict
#: audit must NOT treat it as double consumption.
INIT_MAPPING_AUDIT_EXACT: str = (
    "every source key consumed exactly once except the declared w_v "
    "dual-write alias (backbone.w_v <-> move_decoder.w_v, same tensor); "
    "every target key loaded or fresh-listed with seed; loaded tensors "
    "torch.equal; no empty slots"
)

# ---------------------------------------------------------------------------
# fix-6 C3 + gate-0 A6: parity asset tiers and update provenance.
# Each row pins (tier, run_updates, checkpoint_update): run_updates is the
# source run's total training dose (must equal the tier dose named in the
# tier label); checkpoint_update is the pinned checkpoint_best.pt
# payload's best-inner-val ``update`` (checkpoint_update <= run_updates).
# The legacy ``training_updates`` field stays ONLY as an alias equal to
# ``run_updates`` (single run-dose meaning — never the checkpoint update)
# because the P1 parity-harness lane reads it from the config.
# ---------------------------------------------------------------------------
PARITY_ASSET_TIERS: Dict[str, Tuple[str, int, int]] = {
    "V541_E0_S17": ("v541_e0_runs_final_37900", 37900, 37000),
    "V541_E0_S29": ("v541_e0_runs_final_37900", 37900, 36400),
    "V541_E0_S43": ("v541_e0_runs_final_37900", 37900, 37800),
    "V53MP_C_S17": ("v53mp_20k_s3_formal_15400_auxiliary", 15400, 15400),
    "V53MP_C_S29": ("v53mp_20k_s3_formal_15400_auxiliary", 15400, 15300),
    "V53MP_C_S43": ("v53mp_20k_s3_formal_15400_auxiliary", 15400, 15400),
}
PARITY_TIER_POLICY_FORBIDDEN: Tuple[str, ...] = (
    "smoke_checkpoints",
    "non_pinned_checkpoints",
)
GPU_ATOL = 1.0e-06
GPU_RTOL = 1.0e-05

# ---------------------------------------------------------------------------
# v1.1 frozen initialization contract (ADR-0033 v1.1 §2.4; Oracle findings 4/13)
# ---------------------------------------------------------------------------
INIT_SEED_RECIPE = (
    'seed_init = int(sha256("v542-init|{run_seed}|{init_source_id}")'
    '.digest()[:8], "big") % 2147483647'
)
INIT_SEED_RECIPE_ALGORITHM = (
    "sha256_hexdigest[:16]_as_big_endian_int_mod_2147483647"
)
CONSTRUCTION_ORDER: Tuple[str, ...] = (
    "1_fresh_backbone_modules_seeded_by_seed_init",
    "2_load_R3_mapped_tensors_over_fresh_init_per_init_mapping",
    "3_zero_init_special_parameters_per_candidate_spec",
    "4_direct_move_decoder_with_shared_w_v_from_R3",
    "5_process_head_from_deterministic_anchor_replay",
    "6_set_seed(run_seed)_before_training",
)
INIT_SOURCE_OVERRIDES = {"A1-C": "A1", "A5-C": "A5"}
R3_MAPPING_CANDIDATES = frozenset(
    {"A0", "A1", "A2", "A3", "A4", "A5", "A1-C", "A5-C"}
)

# ---------------------------------------------------------------------------
# v1.1 frozen interface strings (ADR-0033 v1.1 §2.1–2.2; Oracle finding 9)
# ---------------------------------------------------------------------------
A0_CAPTURE_SPEC = (
    "message_mlp[-1] forward hook, exactly once, finally-removed, no detach"
)
W_V_SHARING_SPEC = "backbone.w_v is move_decoder.w_v (same tensor object)"
STATE_DICT_ALIASES = ("backbone.w_v", "move_decoder.w_v")
DECODER_INPUTS = ("edge_msg", "pos", "edge_index", "key_index", "degK")
DECODER_FORBIDDEN = ("absolute_coordinates", "centroid", "orientation",
                     "edge_mlp_recompute")
PROCESS_EXTERNAL_KEYS = ("u", "z", "q", "endpoint_logit", "order_score")
PROCESS_FORWARD_MOVE_KEYS = ("u", "z")

# ---------------------------------------------------------------------------
# v1.1 frozen identity/start contract (ADR-0033 v1.1 §2.6; Oracle findings 10–12)
# ---------------------------------------------------------------------------
IDENTITY_REQUIRED_FIELDS: Tuple[str, ...] = (
    "source_manifest_sha256",
    "config_sha256",
    "common_protocol_sha256",
    "environment_sha256",
    "backbone_spec_sha256",
    "init_mapping_sha256",
    "process_head_anchor_sha256",
    "statistics_contract_sha256",
    "bootstrap_seed_root",
    "plan_sha256",
    "oracle_design_source_sha256",
    "runplan_sha256",
    "budget_policy_sha256",
    "sampler_spec_sha256",
)
IDENTITY_OPTIONAL_FIELDS: Tuple[str, ...] = ("code_commit",)
#: Frozen canonical T1 start gates — EXACTLY the config list (fix-6: the
#: A0 sanity band is a hard T1 precondition, gate id verified against the
#: G–V registry below).
T1_START_REQUIRES: Tuple[str, ...] = (
    "V542-H0-legality",
    "V542-H1-resource",
    "V542-WP1-interface-parity",
    "V542-WP2-candidates",
    "V542-smoke-200",
    "V542-profile",
    "V542-diagnostic-2000",
    "V542-A0-sanity-band",
    "V542-budget-signoff",
    "runplan_pin_record",
)
#: Record requirements (not registry gates) allowed in t1_start_requires.
T1_RECORD_WHITELIST = frozenset({"runplan_pin_record"})
#: Canonical registry gate IDs (every non-record entry must resolve here).
T1_CANONICAL_GATES: Tuple[str, ...] = tuple(
    g for g in T1_START_REQUIRES if g not in T1_RECORD_WHITELIST
)
A0_DECISION_TEMPLATE = (
    "config/campaigns/v542_templates/a0_reuse_decision.template.json"
)
A0_DECISION_INSTANCE = (
    "outputs/pesn/v542_backbone_50k/plan/a0_reuse_decision.json"
)

_DEFAULT_CONFIG = (
    PROJECT_ROOT / "config" / "campaigns" / "pes2ts_v542_backbone_50k.yaml"
)
DEFAULT_CONFIG = _DEFAULT_CONFIG

_GV_DEFAULT = (
    "campaign_id", "primary_g", "supporting_g", "gates", "map_version"
)

_SHA64_RE = re.compile(r"^[0-9a-f]{64}$")


class V542SchemaError(ValueError):
    """Typed rejection with a stable machine code."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Read + minimally check the V542 campaign yaml (fail-closed)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not cfg_path.is_file():
        raise V542SchemaError("V542_CONFIG_MISSING", str(cfg_path))
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V542SchemaError("V542_CONFIG_MISSING", f"{cfg_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise V542SchemaError("V542_CONFIG_MISSING", f"{cfg_path}: not a mapping")
    return raw


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _err(code: str, detail: str) -> None:
    raise V542SchemaError(code, detail)


def _normalize_steps(raw: Any) -> List[str]:
    """Flatten a YAML procedure list whose ``key: value`` rows parse as maps."""
    out: List[str] = []
    for item in raw if isinstance(raw, Sequence) and not isinstance(
            raw, (str, bytes)) else []:
        if isinstance(item, Mapping):
            for key, val in item.items():
                out.append(f"{key}: {val}")
        else:
            out.append(str(item))
    return out


def _is_positive_int(value: Any) -> bool:
    """True for genuine positive ints (bools, strings and floats rejected)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


# ---------------------------------------------------------------------------
# individual checks — each returns None or raises V542SchemaError
# ---------------------------------------------------------------------------
def _check_schema(cfg: Mapping[str, Any]) -> None:
    if cfg.get("schema") != SCHEMA_ID:
        _err("V542_SCHEMA", f"expected {SCHEMA_ID!r}, got {cfg.get('schema')!r}")
    if cfg.get("campaign_id") != CAMPAIGN_ID:
        _err(
            "V542_SCHEMA",
            f"campaign_id expected {CAMPAIGN_ID!r}, got {cfg.get('campaign_id')!r}",
        )


def _check_required_fields(cfg: Mapping[str, Any]) -> None:
    missing = [k for k in REQUIRED_FIELDS if cfg.get(k) is None]
    if missing:
        _err("V542_MISSING_FIELD", f"missing/None required field(s): {missing}")


def _check_objective_and_arms(cfg: Mapping[str, Any]) -> None:
    objective = cfg.get("objective")
    if objective != OBJECTIVE:
        _err(
            "V542_OBJECTIVE",
            f"objective must be {OBJECTIVE!r} (only legal external task name), "
            f"got {objective!r}",
        )
    for key in ARM_MATRIX_KEYS:
        if key in cfg:
            _err(
                "V542_ARM_MATRIX",
                f"top-level {key!r} re-exposes a legacy multi-arm matrix",
            )


def _check_backbone_registry(cfg: Mapping[str, Any]) -> None:
    contract = cfg.get("backbone_contract")
    if not isinstance(contract, Mapping):
        _err("V542_MISSING_FIELD", "backbone_contract must be a mapping")
    registry = contract.get("candidates")
    if not isinstance(registry, Mapping) or not registry:
        _err("V542_MISSING_FIELD", "backbone_contract.candidates missing/empty")
    unknown = sorted(set(registry) - KNOWN_BACKBONES)
    if unknown:
        _err(
            "V542_BACKBONE_UNKNOWN",
            f"backbone id(s) {unknown} outside frozen A0–A5/A1-C/A5-C registry",
        )
    missing = sorted(KNOWN_BACKBONES - set(registry))
    if missing:
        _err(
            "V542_BACKBONE_UNKNOWN",
            f"frozen backbone(s) {missing} missing from the registry",
        )
    for bid, spec in registry.items():
        if not isinstance(spec, Mapping):
            _err("V542_MISSING_FIELD", f"candidates.{bid} must be a mapping")
        for field in ("family", "hidden", "backbone_active", "estimated"):
            if spec.get(field) is None:
                _err("V542_MISSING_FIELD", f"candidates.{bid}.{field} missing")
        exp_family, _exp_main, exp_diag, exp_est = BACKBONE_REGISTRY[bid]
        if spec.get("family") != exp_family:
            _err(
                "V542_BACKBONE_UNKNOWN",
                f"candidates.{bid}.family must be {exp_family!r}, "
                f"got {spec.get('family')!r}",
            )
        if bool(spec.get("estimated")) != exp_est:
            _err(
                "V542_BACKBONE_UNKNOWN",
                f"candidates.{bid}.estimated must be {exp_est} (A0 numbers are "
                "migration invariants; candidates are design estimates)",
            )
        if bool(spec.get("diagnostic_only", False)) != exp_diag:
            _err(
                "V542_BACKBONE_UNKNOWN",
                f"candidates.{bid}.diagnostic_only must be {exp_diag}",
            )
        if spec.get("hidden") != 48:
            _err(
                "V542_BACKBONE_UNKNOWN",
                f"candidates.{bid}.hidden must be 48 (interface width), "
                f"got {spec.get('hidden')!r}",
            )
        # v1.1: A1-C/A5-C init streams are frozen overrides (Oracle finding 4).
        if bid in INIT_SOURCE_OVERRIDES:
            if spec.get("init_source_id") != INIT_SOURCE_OVERRIDES[bid]:
                _err(
                    "V542_INIT_SOURCE",
                    f"candidates.{bid}.init_source_id must be "
                    f"{INIT_SOURCE_OVERRIDES[bid]!r} (frozen init-stream "
                    f"override), got {spec.get('init_source_id')!r}",
                )
            if spec.get("winner_eligible") is not False:
                _err(
                    "V542_BACKBONE_UNKNOWN",
                    f"candidates.{bid}.winner_eligible must be false "
                    "(attribution diagnostics never compete)",
                )
        # gate-2 Ruling 2 (2026-10-30): the per-candidate W_g / GRUCell
        # bias record is part of the frozen parameter arithmetic (288 vs
        # 240 per W_g cell; 14,112 vs 13,824 per GRUCell).  The fields
        # must state the code's truth with EXACT booleans — missing
        # fields, value drift and token-preserving near-misses ("true",
        # 1, ...) are all typed rejections; candidates without a W_g
        # injection or GRU cells may not carry the fields at all.
        if bid in W_G_BIAS:
            if "w_g_bias" not in spec:
                _err(
                    "V542_MISSING_FIELD",
                    f"candidates.{bid}.w_g_bias missing (gate-2 Ruling 2: "
                    "the per-candidate W_g bias choice must be recorded "
                    f"explicitly; expected {W_G_BIAS[bid]})",
                )
            exp_w_g = W_G_BIAS[bid]
            got_w_g = spec.get("w_g_bias")
            if got_w_g is not exp_w_g:
                _err(
                    "V542_PARAM_BUDGET",
                    f"candidates.{bid}.w_g_bias must be exactly the frozen "
                    f"boolean {exp_w_g} (biased nn.Linear(5,48)=288 params "
                    "for A2/A4; nn.Linear(5,48,bias=False)=240 params per "
                    "cell for A5/A5-C; gate-2 Ruling 2), got "
                    f"{got_w_g!r} ({type(got_w_g).__name__}) — string/int "
                    "near-misses such as 'true'/1 are rejected",
                )
        else:
            if "w_g_bias" in spec:
                _err(
                    "V542_PARAM_BUDGET",
                    f"candidates.{bid} has no W_g graph-conditioning "
                    "injection — w_g_bias must not be recorded (gate-2 "
                    "Ruling 2 covers A2/A4/A5/A5-C only)",
                )
        if bid in GRU_CELL_BIAS:
            if "gru_cell_bias" not in spec:
                _err(
                    "V542_MISSING_FIELD",
                    f"candidates.{bid}.gru_cell_bias missing (gate-2 "
                    "Ruling 2: the per-GRUCell bias record must be "
                    f"explicit and in cell order; expected "
                    f"{list(GRU_CELL_BIAS[bid])})",
                )
            exp_cells = GRU_CELL_BIAS[bid]
            got_cells = spec.get("gru_cell_bias")
            if not isinstance(got_cells, Sequence) or isinstance(
                    got_cells, (str, bytes)) \
                    or len(got_cells) != len(exp_cells) \
                    or any(g is not e for g, e in zip(got_cells, exp_cells)):
                _err(
                    "V542_PARAM_BUDGET",
                    f"candidates.{bid}.gru_cell_bias must be exactly "
                    f"{list(exp_cells)} in cell order (cell 0 = A5 recipe "
                    "biased GRUCell(48,48)=14112; A5-C cell 1 = "
                    "GRUCell(48,48,bias=False)=13824; gate-2 Ruling 2), "
                    f"got {got_cells!r} — only exact booleans accepted",
                )
        else:
            if "gru_cell_bias" in spec:
                _err(
                    "V542_PARAM_BUDGET",
                    f"candidates.{bid} has no GRU cells — gru_cell_bias "
                    "must not be recorded (gate-2 Ruling 2 covers A5/A5-C "
                    "only)",
                )
    widths = contract.get("candidate_common_widths")
    if not isinstance(widths, Mapping):
        _err(
            "V542_MISSING_FIELD", "backbone_contract.candidate_common_widths missing")
    if dict(widths) != {"hidden": 48, "z": Z_DIM, "edge_msg": EDGE_MSG_DIM,
                        "node_input": 21}:
        _err(
            "V542_BACKBONE_UNKNOWN",
            f"candidate_common_widths must be hidden48/z64/edge_msg48/node_input21, "
            f"got {widths!r}",
        )


def _check_forward_view(cfg: Mapping[str, Any]) -> None:
    contract = cfg.get("backbone_contract")
    keys = contract.get("forward_view_keys") if isinstance(contract, Mapping) else None
    if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)) or not keys:
        _err(
            "V542_MISSING_FIELD", "backbone_contract.forward_view_keys missing/empty"
        )
    whitelist = set(FORWARD_VIEW_KEYS_V3)
    illegal = sorted({str(k) for k in keys} - whitelist)
    if illegal:
        _err(
            "V542_FORWARD_KEY",
            f"forward key(s) {illegal} not in contract whitelist FORWARD_VIEW_KEYS_V3",
        )
    forbidden = contract.get("forbidden_input_classes")
    if not isinstance(forbidden, Sequence) or isinstance(forbidden, (str, bytes)) \
            or not forbidden:
        _err(
            "V542_MISSING_FIELD",
            "backbone_contract.forbidden_input_classes missing/empty",
        )


def _check_interface(cfg: Mapping[str, Any]) -> None:
    """BackboneOutput / Decoder / PROCESS / A0-migration / parity contract.

    v1.1 hardening (Oracle finding 9): every interface subfield, the exact
    ``w_v`` same-tensor statement, the capture-hook spec and the six pinned
    parity assets are enforced — no substring-only checks for THOSE items.
    Note (gate-1 C5 correction): ``a0_migration.core_loop`` remains a
    substring check (requires the ``verbatim`` token; ``"not verbatim"``
    currently passes) — converting it to an exact frozen string is open
    gate-0 advisory A2 (close before P3/WP6).
    """
    contract = cfg.get("backbone_contract")
    iface = contract.get("interface") if isinstance(contract, Mapping) else None
    if not isinstance(iface, Mapping):
        _err("V542_MISSING_FIELD", "backbone_contract.interface missing")
    if tuple(iface.get("backbone_output_fields") or ()) != ("z", "edge_msg"):
        _err(
            "V542_INTERFACE",
            "BackboneOutput must expose exactly the two functional fields "
            f"(z, edge_msg), got {iface.get('backbone_output_fields')!r}",
        )
    z_block = iface.get("z")
    if not isinstance(z_block, Mapping) or list(z_block.get("shape") or []) != ["G", Z_DIM]:
        _err(
            "V542_INTERFACE", "interface.z.shape must be [G, 64]")
    if z_block.get("tanh_saturated") is not True:
        _err(
            "V542_INTERFACE",
            "interface.z.tanh_saturated must be true (z is the tanh-saturated "
            "graph embedding)",
        )
    if z_block.get("order") != "batch_graph_index":
        _err(
            "V542_INTERFACE",
            f"interface.z.order must be 'batch_graph_index', "
            f"got {z_block.get('order')!r}",
        )
    if str(z_block.get("dtype") or "") != "fp32":
        _err(
            "V542_INTERFACE", f"interface.z.dtype must be fp32, "
            f"got {z_block.get('dtype')!r}")
    em_block = iface.get("edge_msg")
    if not isinstance(em_block, Mapping) or list(em_block.get("shape") or []) != ["E", EDGE_MSG_DIM]:
        _err(
            "V542_INTERFACE", "interface.edge_msg.shape must be [E, 48]")
    if em_block.get("last_layer") is not True:
        _err(
            "V542_INTERFACE",
            "interface.edge_msg.last_layer must be true (edge_msg is the last "
            "message layer output)",
        )
    if em_block.get("autograd") is not True:
        _err(
            "V542_INTERFACE",
            "interface.edge_msg.autograd must be true (gradients must flow "
            "through edge_msg into the backbone)",
        )
    if em_block.get("order") != "input_edge_index_columns":
        _err(
            "V542_INTERFACE",
            "interface.edge_msg.order must be 'input_edge_index_columns' "
            f"(row i corresponds to edge_index column i), got "
            f"{em_block.get('order')!r}",
        )
    if str(em_block.get("dtype") or "") != "fp32":
        _err(
            "V542_INTERFACE", f"interface.edge_msg.dtype must be fp32, "
            f"got {em_block.get('dtype')!r}")
    if iface.get("u_in_interface") is not False or iface.get("node_h_in_interface") is not False:
        _err(
            "V542_INTERFACE",
            "u/node_h must stay out of BackboneOutput (u from the shared "
            "Decoder; node_h via diagnostic hook only)",
        )

    decoder = contract.get("move_decoder")
    if not isinstance(decoder, Mapping):
        _err(
            "V542_MISSING_FIELD", "backbone_contract.move_decoder missing")
    if decoder.get("class") != "DirectMoveDecoder":
        _err(
            "V542_INTERFACE",
            f"move_decoder.class must be DirectMoveDecoder, got {decoder.get('class')!r}",
        )
    if decoder.get("trainable_params") != DECODER_PARAMS or \
            decoder.get("only_parameter") != "w_v":
        _err(
            "V542_INTERFACE",
            "DirectMoveDecoder must have exactly 49 params (only w_v); "
            f"got trainable_params={decoder.get('trainable_params')!r} "
            f"only_parameter={decoder.get('only_parameter')!r}",
        )
    if float(decoder.get("a_ref_angstrom") or 0) != 0.08:
        _err(
            "V542_INTERFACE",
            f"move_decoder.a_ref_angstrom must be 0.08 (teacher action "
            f"scale), got {decoder.get('a_ref_angstrom')!r}",
        )
    if decoder.get("a_ref_learnable") is not False:
        _err(
            "V542_INTERFACE", "a_ref must stay a non-learnable constant (0.08 A)")
    if decoder.get("reads_supervision_keys") is not False:
        _err(
            "V542_INTERFACE",
            "Decoder consumes 5 tensors only — supervision-key access forbidden",
        )
    inputs = tuple(decoder.get("inputs") or ())
    if inputs != DECODER_INPUTS:
        _err(
            "V542_INTERFACE",
            f"move_decoder.inputs must be the frozen 5-tuple, got {inputs!r}",
        )
    if decoder.get("output_order") != "key_index_order":
        _err(
            "V542_INTERFACE",
            "move_decoder.output_order must be 'key_index_order' (decoder "
            "output aligns to key_index; no batch readout), got "
            f"{decoder.get('output_order')!r}",
        )
    if decoder.get("non_key_policy") != "structural_zero_rows":
        _err(
            "V542_INTERFACE",
            "move_decoder.non_key_policy must be 'structural_zero_rows' "
            "(non-key displacement is structurally zero), got "
            f"{decoder.get('non_key_policy')!r}",
        )
    forbidden = decoder.get("forbidden")
    if not isinstance(forbidden, Sequence) or isinstance(forbidden, (str, bytes)) \
            or set(str(x) for x in forbidden) != set(DECODER_FORBIDDEN):
        _err(
            "V542_INTERFACE",
            f"move_decoder.forbidden must be exactly {list(DECODER_FORBIDDEN)}, "
            f"got {forbidden!r}",
        )

    head = contract.get("process_head")
    if not isinstance(head, Mapping):
        _err(
            "V542_MISSING_FIELD", "backbone_contract.process_head missing")
    if head.get("class") != "ReactionProcessHeadV53MP" or \
            head.get("params") != PROCESS_PARAMS:
        _err(
            "V542_INTERFACE",
            "PROCESS head must stay ReactionProcessHeadV53MP (2178 params), got "
            f"{head.get('class')!r}/{head.get('params')!r}",
        )
    if tuple(head.get("reads") or ()) != ("z",):
        _err(
            "V542_INTERFACE",
            f"process_head.reads must be exactly ['z'] (never edge_msg), "
            f"got {head.get('reads')!r}",
        )
    if head.get("reads_edge_msg") is not False:
        _err(
            "V542_INTERFACE",
            "process_head.reads_edge_msg must be false (PROCESS consumes z only)",
        )
    if head.get("order_score_is_q") is not True:
        _err(
            "V542_INTERFACE", "order_score IS q must be preserved")
    if tuple(head.get("external_output_keys") or ()) != PROCESS_EXTERNAL_KEYS:
        _err(
            "V542_INTERFACE",
            "external output keys must stay the frozen 5-key set "
            f"{head.get('external_output_keys')!r}",
        )
    if tuple(head.get("forward_move_keys") or ()) != PROCESS_FORWARD_MOVE_KEYS:
        _err(
            "V542_INTERFACE",
            "forward_move keys must stay (u, z)")
    if head.get("new_external_keys_forbidden") is not True:
        _err(
            "V542_INTERFACE",
            "process_head.new_external_keys_forbidden must be true (trainer/eval "
            "depend on exactly the frozen 5-key external output)",
        )

    a0 = contract.get("a0_migration")
    if not isinstance(a0, Mapping):
        _err(
            "V542_MISSING_FIELD", "backbone_contract.a0_migration missing")
    if a0.get("subclass_of") != "StarEGNNV3":
        _err(
            "V542_INTERFACE",
            "A0 must be a StarEGNNV3 subclass adapter (no legacy edits), got "
            f"{a0.get('subclass_of')!r}",
        )
    if "verbatim" not in str(a0.get("core_loop") or ""):
        _err(
            "V542_INTERFACE",
            "a0_migration.core_loop must keep super().forward verbatim",
        )
    if a0.get("capture") != A0_CAPTURE_SPEC:
        _err(
            "V542_INTERFACE",
            "a0_migration.capture must be exactly "
            f"{A0_CAPTURE_SPEC!r}, got {a0.get('capture')!r}",
        )
    if a0.get("w_v_sharing") != W_V_SHARING_SPEC:
        _err(
            "V542_INTERFACE",
            "A0 w_v must be the SAME tensor shared between backbone and "
            f"decoder — exact frozen statement {W_V_SHARING_SPEC!r} required, "
            f"got {a0.get('w_v_sharing')!r} (substring claims such as "
            "'not same tensor' are rejected)",
        )
    if tuple(a0.get("state_dict_aliases") or ()) != STATE_DICT_ALIASES:
        _err(
            "V542_INTERFACE",
            f"a0_migration.state_dict_aliases must be {list(STATE_DICT_ALIASES)}, "
            f"got {a0.get('state_dict_aliases')!r}",
        )
    if a0.get("alias_assertion") != "torch_equal_on_save_and_load":
        _err(
            "V542_INTERFACE",
            "a0_migration.alias_assertion must be 'torch_equal_on_save_and_load' "
            f"(alias tensors must be torch.equal on save AND load), got "
            f"{a0.get('alias_assertion')!r}",
        )
    if a0.get("mapping") != "explicit_strict_audit":
        _err(
            "V542_INTERFACE",
            "a0_migration.mapping must be 'explicit_strict_audit' "
            f"(strict=True only; no silent key filtering), got "
            f"{a0.get('mapping')!r}",
        )
    if a0.get("torch_compile") is not False:
        _err(
            "V542_INTERFACE",
            "torch.compile is forbidden in V5.4.2 (hook/parity failure)",
        )
    parity = a0.get("parity")
    if not isinstance(parity, Mapping) or parity.get("cpu") != "bitwise":
        _err(
            "V542_INTERFACE", "A0 parity.cpu must be 'bitwise'")
    if float(parity.get("gpu_atol") or 0) != 1e-6 or \
            float(parity.get("gpu_rtol") or 0) != 1e-5:
        _err(
            "V542_INTERFACE",
            f"A0 GPU parity tolerance must be atol=1e-6/rtol=1e-5, got "
            f"{parity.get('gpu_atol')!r}/{parity.get('gpu_rtol')!r}",
        )
    if parity.get("real_data_min_inner_val_batches") != 8:
        _err(
            "V542_INTERFACE",
            "A0 parity must cover >=8 real inner-val batches "
            f"(real_data_min_inner_val_batches=8), got "
            f"{parity.get('real_data_min_inner_val_batches')!r}",
        )
    if parity.get("q_equals_order_score") is not True:
        _err(
            "V542_INTERFACE",
            "A0 parity must assert q_equals_order_score=true",
        )
    if not str(parity.get("evaluator_parity") or "").startswith(
            "recompute_H2_H3_vs_V541"):
        _err(
            "V542_INTERFACE",
            "A0 parity.evaluator_parity must recompute H2/H3 metrics vs the "
            f"V541 archive (bitwise or <=1e-9), got "
            f"{parity.get('evaluator_parity')!r}",
        )
    if parity.get("tf32_during_parity") is not False:
        _err(
            "V542_INTERFACE",
            "TF32 must be disabled during parity (tf32_during_parity=false)",
        )
    if parity.get("required_assets") != PARITY_ASSET_COUNT:
        _err(
            "V542_INTERFACE",
            f"parity.required_assets must be {PARITY_ASSET_COUNT} (6/6 "
            f"required), got {parity.get('required_assets')!r}",
        )
    if not str(parity.get("smoke_path_forbidden") or "").strip():
        _err(
            "V542_INTERFACE",
            "parity.smoke_path_forbidden must state that smoke checkpoints are "
            "never parity assets",
        )
    assets = parity.get("assets")
    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes)) \
            or len(assets) != PARITY_ASSET_COUNT:
        _err(
            "V542_INTERFACE",
            f"parity.assets must list exactly {PARITY_ASSET_COUNT} pinned "
            f"checkpoints, got {assets!r}",
        )
    for idx, (asset, (exp_id, exp_path, exp_sha)) in enumerate(
            zip(assets, PARITY_ASSETS)):
        if not isinstance(asset, Mapping):
            _err("V542_INTERFACE", f"parity.assets[{idx}] must be a mapping")
        aid = asset.get("id")
        if aid != exp_id:
            _err(
                "V542_INTERFACE",
                f"parity.assets[{idx}].id must be {exp_id!r}, got {aid!r}",
            )
        apath = asset.get("path")
        if apath != exp_path:
            _err(
                "V542_INTERFACE",
                f"parity asset {exp_id}: path must be exactly {exp_path!r}, "
                f"got {apath!r} (pin the tiered asset: V541 E0 rows = runs/ "
                "37,900-update run best inner-val checkpoint (checkpoint "
                "update 37000/36400/37800); V53MP C rows = 20k S3 formal "
                "15,400-update checkpoint (best inner-val from a "
                "15,400-update run; checkpoint update 15400/15300/15400); "
                "smoke/non-pinned paths are forbidden)",
            )
        if PARITY_SMOKE_PATH_TOKEN in str(apath):
            _err(
                "V542_INTERFACE",
                f"parity asset {exp_id}: smoke checkpoint path "
                f"{apath!r} is never a valid parity asset",
            )
        asha = asset.get("sha256")
        if asha != exp_sha:
            _err(
                "V542_INTERFACE",
                f"parity asset {exp_id}: sha256 must be exactly {exp_sha!r}, "
                f"got {asha!r}",
            )
        if asset.get("required") is not True:
            _err(
                "V542_INTERFACE",
                f"parity asset {exp_id}: required must be true (6/6 assets are "
                "mandatory — no optional rows)",
            )
        # fix-6 C3 + gate-0 A6: every row carries BOTH provenance numbers
        # so a harness can never mistake the V53MP C 20k/15,400 asset for
        # a 37,900-run checkpoint (or substitute a non-pinned checkpoint):
        # run_updates = the source run's total training dose (must match
        # the tier dose), checkpoint_update = the payload's best-checkpoint
        # `update` (positive int, <= run_updates).  training_updates stays
        # ONLY as an alias equal to run_updates (single run-dose meaning;
        # never the checkpoint update — the P1 parity harness reads it).
        exp_tier, exp_run_updates, exp_checkpoint_update = PARITY_ASSET_TIERS[
            exp_id]
        if asset.get("source_tier") != exp_tier:
            _err(
                "V542_INTERFACE",
                f"parity asset {exp_id}: source_tier must be "
                f"{exp_tier!r}, got {asset.get('source_tier')!r}",
            )
        run_updates = asset.get("run_updates")
        if not _is_positive_int(run_updates):
            _err(
                "V542_INTERFACE",
                f"parity asset {exp_id}: run_updates must be a positive int "
                f"(the source-run training dose), got {run_updates!r}",
            )
        if run_updates != exp_run_updates:
            _err(
                "V542_INTERFACE",
                f"parity asset {exp_id}: run_updates must equal the "
                f"{exp_tier!r} tier dose {exp_run_updates}, got "
                f"{run_updates!r}",
            )
        checkpoint_update = asset.get("checkpoint_update")
        if not _is_positive_int(checkpoint_update):
            _err(
                "V542_INTERFACE",
                f"parity asset {exp_id}: checkpoint_update must be a "
                f"positive int (the payload's best-checkpoint `update`), got "
                f"{checkpoint_update!r}",
            )
        if checkpoint_update > run_updates:
            _err(
                "V542_INTERFACE",
                f"parity asset {exp_id}: checkpoint_update "
                f"{checkpoint_update} must be <= run_updates {run_updates} "
                "(a best checkpoint cannot postdate its own run)",
            )
        if checkpoint_update != exp_checkpoint_update:
            _err(
                "V542_INTERFACE",
                f"parity asset {exp_id}: checkpoint_update must be the "
                f"pinned payload update {exp_checkpoint_update}, got "
                f"{checkpoint_update!r}",
            )
        if asset.get("training_updates") != run_updates:
            _err(
                "V542_INTERFACE",
                f"parity asset {exp_id}: training_updates must equal "
                f"run_updates {run_updates} (single run-dose meaning; the "
                "checkpoint update belongs in checkpoint_update), got "
                f"{asset.get('training_updates')!r}",
            )

    tier_policy = parity.get("asset_tier_policy")
    if not isinstance(tier_policy, Mapping):
        _err(
            "V542_INTERFACE",
            "parity.asset_tier_policy must pin the two-tier asset policy "
            "(V541 E0 runs/ 37,900 main evidence + V53MP C 20k S3 formal "
            "15,400 auxiliary evidence)",
        )
    # gate-0 A6: the per-tier policy text must state the run dose AND the
    # pinned checkpoint-update triple (e.g. 37000/36400/37800).
    tier_run = sorted({(t, u) for t, u, _c in PARITY_ASSET_TIERS.values()})
    tier_ckpts: Dict[str, List[int]] = {}
    for t, _u, c in PARITY_ASSET_TIERS.values():
        tier_ckpts.setdefault(t, []).append(c)
    for tier, run_updates in tier_run:
        if tier not in tier_policy:
            _err(
                "V542_INTERFACE",
                f"parity.asset_tier_policy must describe tier {tier!r}",
            )
        text = str(tier_policy.get(tier) or "")
        if str(run_updates) not in text.replace(",", ""):
            _err(
                "V542_INTERFACE",
                f"parity.asset_tier_policy.{tier} must state the "
                f"{run_updates}-update count, got {text!r}",
            )
        # seed order (S17/S29/S43) as pinned in PARITY_ASSET_TIERS
        ckpt_token = "/".join(str(c) for c in tier_ckpts[tier])
        if ckpt_token not in text.replace(",", ""):
            _err(
                "V542_INTERFACE",
                f"parity.asset_tier_policy.{tier} must state the pinned "
                f"checkpoint updates {ckpt_token} (gate-0 A6 provenance), "
                f"got {text!r}",
            )
    if tuple(tier_policy.get("forbidden") or ()) != \
            PARITY_TIER_POLICY_FORBIDDEN:
        _err(
            "V542_INTERFACE",
            "parity.asset_tier_policy.forbidden must be exactly "
            f"{list(PARITY_TIER_POLICY_FORBIDDEN)} (smoke AND non-pinned "
            f"checkpoints), got {tier_policy.get('forbidden')!r}",
        )
    smoke_forbid = str(parity.get("smoke_path_forbidden") or "")
    if "non-pinned" not in smoke_forbid and "非钉死" not in smoke_forbid:
        _err(
            "V542_INTERFACE",
            "parity.smoke_path_forbidden must forbid smoke AND non-pinned "
            f"checkpoints, got {smoke_forbid!r}",
        )
    # GPU hard cap semantics live in gates.parity.rule (tolerance values are
    # asserted above): floor>cap is INCONCLUSIVE, never a relaxation.
    h1_rule = str((cfg.get("gates") or {}).get("parity", {}).get("rule")
                  if isinstance((cfg.get("gates") or {}).get("parity"), Mapping)
                  else "")
    for token in ("1e-6/1e-5", "floor>cap", "inconclusive"):
        if token not in h1_rule:
            _err(
                "V542_INTERFACE",
                f"gates.parity.rule must carry the GPU hard-cap token "
                f"{token!r} (hard cap / floor>cap => inconclusive), got "
                f"{h1_rule!r}",
            )


def _check_param_budget(cfg: Mapping[str, Any]) -> None:
    contract = cfg.get("backbone_contract")
    budget = contract.get("param_budget") if isinstance(contract, Mapping) else None
    if not isinstance(budget, Mapping):
        _err(
            "V542_MISSING_FIELD", "backbone_contract.param_budget missing")
    if budget.get("backbone_active_max") != BACKBONE_ACTIVE_MAX:
        _err(
            "V542_PARAM_BUDGET",
            f"param_budget.backbone_active_max must be {BACKBONE_ACTIVE_MAX}",
        )
    if budget.get("decoder_params") != DECODER_PARAMS or \
            budget.get("process_params") != PROCESS_PARAMS:
        _err(
            "V542_PARAM_BUDGET",
            "param_budget must keep decoder=49 and process=2178",
        )
    registry = contract.get("candidates") or {}
    a0 = registry.get("A0")
    if not isinstance(a0, Mapping):
        _err("V542_BACKBONE_UNKNOWN", "candidates.A0 missing")
    if a0.get("backbone_active") != A0_ACTIVE_PARAMS or \
            a0.get("joint") != A0_JOINT_PARAMS:
        _err(
            "V542_PARAM_BUDGET",
            "A0 migration invariant: backbone_active must be 55392 and joint "
            f"57619, got {a0.get('backbone_active')!r}/{a0.get('joint')!r} — "
            "check w_v double-count or z_head drift",
        )
    for bid, spec in registry.items():
        if not isinstance(spec, Mapping):
            continue
        active = spec.get("backbone_active")
        if isinstance(active, int) and active > BACKBONE_ACTIVE_MAX:
            _err(
                "V542_PARAM_BUDGET",
                f"candidates.{bid}.backbone_active {active} exceeds the frozen "
                f"cap {BACKBONE_ACTIVE_MAX} — simplify before results, never after",
            )
    a0_mig = contract.get("a0_migration") or {}
    if a0_mig.get("backbone_active_params") != A0_ACTIVE_PARAMS or \
            a0_mig.get("joint_params") != A0_JOINT_PARAMS:
        _err(
            "V542_PARAM_BUDGET",
            "a0_migration must restate the 55392/57619 invariants",
        )


def _check_init_source(cfg: Mapping[str, Any], *, formal: bool) -> List[str]:
    """R3-only warm start + v1.1 frozen init policy (Oracle §3, findings 4/9/13).

    The R3 path must equal ``R3_REL_PATH`` and the SHA must equal the frozen
    ``R3_SHA256`` constant exactly — substring checks are not sufficient.
    """
    tc = cfg.get("training_contract")
    init = tc.get("backbone_init") if isinstance(tc, Mapping) else None
    if not isinstance(init, Mapping):
        _err(
            "V542_MISSING_FIELD", "training_contract.backbone_init missing")
    path = init.get("path")
    sha = init.get("sha256")
    if path != R3_REL_PATH:
        _err(
            "V542_INIT_SOURCE",
            "warm-start source must be exactly the frozen R3 checkpoint path "
            f"{R3_REL_PATH!r}; got {path!r} — V541 E0 final is parity-only and "
            "no alternate R3 copy may substitute",
        )
    if not isinstance(sha, str) or not _SHA64_RE.match(sha) or sha == "0" * 64:
        _err(
            "V542_INIT_SOURCE",
            "backbone_init.sha256 must be a non-zero 64-hex digest, got "
            f"{sha!r}",
        )
    if sha != R3_SHA256:
        _err(
            "V542_R3_INIT",
            f"backbone_init.sha256 {sha[:12]}... != frozen R3_SHA256 "
            f"{R3_SHA256[:12]}... — restore the checkpoint or re-sign the "
            "config (the validator compares against the frozen constant, not "
            "the config copy)",
        )
    if init.get("run_id") != R3_RUN_ID:
        _err(
            "V542_INIT_SOURCE",
            f"backbone_init.run_id must be {R3_RUN_ID!r}, got {init.get('run_id')!r}",
        )
    if init.get("verify_on_validate") is not True:
        _err(
            "V542_INIT_SOURCE", "backbone_init.verify_on_validate must be true")
    forbidden = init.get("forbidden_sources")
    if not isinstance(forbidden, Sequence) or isinstance(forbidden, (str, bytes)) \
            or "V541_E0_final" not in set(str(x) for x in forbidden):
        _err(
            "V542_INIT_SOURCE",
            "backbone_init.forbidden_sources must explicitly name V541_E0_final",
        )
    blocked: List[str] = []
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    if not target.is_file():
        if formal:
            _err(
                "V542_R3_INIT",
                f"R3 init checkpoint {path} is missing — a formal run cannot "
                "start from an unverifiable init",
            )
        blocked.append(
            f"backbone_init file missing: {path} (R3 SHA cannot be re-verified)")
        return blocked
    actual = _sha256_of_file(target)
    if actual != R3_SHA256:
        _err(
            "V542_R3_INIT",
            f"R3 init {path} hashes to {actual[:12]} but the frozen digest is "
            f"{R3_SHA256[:12]} — restore the checkpoint or re-sign the config",
        )

    policy = tc.get("init_policy")
    if not isinstance(policy, Mapping):
        _err(
            "V542_MISSING_FIELD", "training_contract.init_policy missing")
    mapping = str(policy.get("mapping") or "")
    if "explicit" not in mapping or "shape" not in mapping:
        _err(
            "V542_INIT_SOURCE",
            "init_policy.mapping must freeze the explicit semantic mapping and "
            "forbid runtime shape heuristics",
        )
    # gate-0 A8: the mapping-audit string is frozen EXACTLY (not by token
    # presence) so the declared w_v dual-write alias stays a legitimate
    # exception instead of looking like double consumption.
    audit = str(policy.get("mapping_audit") or "")
    if audit != INIT_MAPPING_AUDIT_EXACT:
        _err(
            "V542_INIT_SOURCE",
            "init_policy.mapping_audit must be exactly the frozen audit "
            f"string with the declared w_v dual-write alias exception "
            f"{INIT_MAPPING_AUDIT_EXACT!r}, got {audit!r}",
        )
    fresh = policy.get("candidate_fresh_init")
    if not isinstance(fresh, Mapping):
        _err(
            "V542_MISSING_FIELD",
            "training_contract.init_policy.candidate_fresh_init missing",
        )
    if fresh.get("seed_recipe") != INIT_SEED_RECIPE:
        _err(
            "V542_INIT_SOURCE",
            "candidate_fresh_init.seed_recipe must be the frozen sha256 "
            "first-8-bytes-big-endian mod 2147483647 recipe "
            f"{INIT_SEED_RECIPE!r}, got {fresh.get('seed_recipe')!r}",
        )
    if fresh.get("seed_recipe_algorithm") != INIT_SEED_RECIPE_ALGORITHM:
        _err(
            "V542_INIT_SOURCE",
            "candidate_fresh_init.seed_recipe_algorithm must be "
            f"{INIT_SEED_RECIPE_ALGORITHM!r}, got "
            f"{fresh.get('seed_recipe_algorithm')!r}",
        )
    if fresh.get("init_source_id_default") != "candidate_id":
        _err(
            "V542_INIT_SOURCE",
            "candidate_fresh_init.init_source_id_default must be 'candidate_id'",
        )
    overrides = dict(fresh.get("init_source_id_overrides") or {})
    if overrides != INIT_SOURCE_OVERRIDES:
        _err(
            "V542_INIT_SOURCE",
            "candidate_fresh_init.init_source_id_overrides must be exactly "
            f"{INIT_SOURCE_OVERRIDES} (A1-C shares A1's init stream, A5-C "
            f"shares A5's), got {overrides!r}",
        )
    order = tuple(policy.get("construction_order") or ())
    if order != CONSTRUCTION_ORDER:
        _err(
            "V542_INIT_SOURCE",
            "init_policy.construction_order must be the frozen six-step order "
            f"{list(CONSTRUCTION_ORDER)}, got {list(order)!r}",
        )
    if not str(policy.get("construction_rng") or "").strip():
        _err(
            "V542_INIT_SOURCE",
            "init_policy.construction_rng must document the fork_rng isolation "
            "of model construction",
        )
    if policy.get("reset_seed_before_training") is not True:
        _err(
            "V542_INIT_SOURCE",
            "init_policy.reset_seed_before_training must be true "
            "(construction must not perturb the training RNG stream)",
        )
    anchor = policy.get("process_head_anchor")
    if not isinstance(anchor, Mapping) or "fork_rng" not in str(anchor.get("method") or ""):
        _err(
            "V542_INIT_SOURCE",
            "init_policy.process_head_anchor must use the fork_rng replay "
            "method (common deterministic head init across candidates)",
        )
    if tuple(anchor.get("seeds") or ()) != (17, 29, 43):
        _err(
            "V542_INIT_SOURCE",
            "process_head_anchor.seeds must be (17, 29, 43)",
        )
    if not str(anchor.get("artifact") or "").strip():
        _err(
            "V542_INIT_SOURCE",
            "process_head_anchor.artifact must pin the anchor replay record "
            "path (algorithm + seeds + state_dict digest)",
        )
    if anchor.get("included_in_run_identity") is not True:
        _err(
            "V542_INIT_SOURCE",
            "process_head_anchor.included_in_run_identity must be true",
        )
    if anchor.get("excludes_w_v") is not True:
        _err(
            "V542_INIT_SOURCE",
            "process_head_anchor.excludes_w_v must be true (w_v maps from R3 "
            "and is identical across candidates)",
        )
    init_mapping = policy.get("init_mapping")
    if not isinstance(init_mapping, Mapping):
        _err(
            "V542_MISSING_FIELD", "init_policy.init_mapping missing")
    note = str(init_mapping.get("note") or "")
    # gate-0 mapping-count clarification: the note must state the exact
    # counts — 37 backbone keys identity (with the w_v dual-write alias)
    # for the backbone-only R3 mapping, and 37 + 6 process-head keys = 43
    # joint keys for joint-checkpoint strict audits (loader.py maps all 43).
    for token in ("w_v", "R3", "37", "43"):
        if token not in note:
            _err(
                "V542_INIT_SOURCE",
                "init_policy.init_mapping.note must state that every "
                "candidate's w_v maps from R3 and that A0 identity is 37 "
                "backbone keys (w_v dual-write alias) + 6 process-head keys "
                f"= 43 joint keys for joint checkpoints (token {token!r} "
                f"missing), got {note!r}",
            )
    semantic = init_mapping.get("r3_semantic_mapping")
    if not isinstance(semantic, Mapping) or set(semantic) != R3_MAPPING_CANDIDATES:
        _err(
            "V542_INIT_SOURCE",
            "init_mapping.r3_semantic_mapping must define a per-candidate "
            f"{{target: source}} table for exactly {sorted(R3_MAPPING_CANDIDATES)}, "
            f"got {sorted(semantic) if isinstance(semantic, Mapping) else semantic!r}",
        )
    if "V_0=0" not in str(semantic.get("A2") or ""):
        _err(
            "V542_INIT_SOURCE",
            "r3_semantic_mapping.A2 must freeze V_0=0 (Oracle finding 13: the "
            "A2 vector-channel seed must not be a post-hoc choice)",
        )
    if "A1 init stream" not in str(semantic.get("A1-C") or ""):
        _err(
            "V542_INIT_SOURCE",
            "r3_semantic_mapping.A1-C must state the third layer uses A1's "
            "init stream (init_source_id=A1)",
        )
    if "A5 init stream" not in str(semantic.get("A5-C") or ""):
        _err(
            "V542_INIT_SOURCE",
            "r3_semantic_mapping.A5-C must state the second GRUCell uses A5's "
            "init stream (init_source_id=A5)",
        )
    zero = policy.get("per_candidate_zero_init")
    if not isinstance(zero, Mapping):
        _err(
            "V542_MISSING_FIELD", "init_policy.per_candidate_zero_init missing")
    for bid, tokens in (("A1", ("W_d", "零初始化")), ("A3", ("W_o", "零初始化")),
                        ("A4", ("零初始化",)), ("A5", ("W_g", "零初始化"))):
        text = str(zero.get(bid) or "")
        for token in tokens:
            if token not in text:
                _err(
                    "V542_INIT_SOURCE",
                    f"per_candidate_zero_init.{bid} must freeze the {token} "
                    f"zero-init choice, got {text!r}",
                )
    a2_zero = zero.get("A2")
    if not isinstance(a2_zero, Mapping) or a2_zero.get("V_0") != 0 or \
            a2_zero.get("phi_u_phi_g_final_output") != "zero_init" or \
            a2_zero.get("W_g") != "zero":
        _err(
            "V542_INIT_SOURCE",
            "per_candidate_zero_init.A2 must freeze V_0=0, "
            "phi_u_phi_g_final_output=zero_init and W_g=zero (Oracle "
            f"finding 13), got {a2_zero!r}",
        )
    groups = policy.get("param_groups")
    if not isinstance(groups, Mapping) or groups.get("method") != "by_id":
        _err(
            "V542_INIT_SOURCE",
            "init_policy.param_groups.method must be 'by_id' — the V541 "
            "name-prefix classifier cannot classify move_decoder.*",
        )
    core_group = groups.get("core")
    if not isinstance(core_group, Mapping) or \
            list(core_group.get("members") or ()) != ["backbone", "move_decoder"] or \
            float(core_group.get("lr") or 0) != 1e-4:
        _err(
            "V542_INIT_SOURCE",
            "param_groups.core must be members=[backbone, move_decoder] with "
            f"lr=1.0e-4, got {core_group!r}",
        )
    head_group = groups.get("head")
    if not isinstance(head_group, Mapping) or \
            list(head_group.get("members") or ()) != ["process_head"] or \
            float(head_group.get("lr") or 0) != 3e-4:
        _err(
            "V542_INIT_SOURCE",
            "param_groups.head must be members=[process_head] with lr=3.0e-4, "
            f"got {head_group!r}",
        )
    if groups.get("w0") != "core frozen, head live" or \
            groups.get("w1") != "all joint":
        _err(
            "V542_INIT_SOURCE",
            "param_groups must keep w0='core frozen, head live' and "
            f"w1='all joint', got w0={groups.get('w0')!r} "
            f"w1={groups.get('w1')!r}",
        )
    return blocked


def _check_a0_decision(cfg: Mapping[str, Any]) -> None:
    a0 = cfg.get("a0_baseline")
    if not isinstance(a0, Mapping):
        _err("V542_MISSING_FIELD", "a0_baseline must be a mapping")
    if a0.get("decision") != A0_DECISION:
        _err(
            "V542_A0_REUSE",
            f"a0_baseline.decision must be {A0_DECISION!r} (frozen WP0 ruling: "
            f"new interface / construction RNG / param-group / evaluation "
            f"identity make V541 E0 reuse unsignable), got {a0.get('decision')!r}",
        )
    if a0.get("shadow_run_required") is not False:
        _err(
            "V542_A0_REUSE",
            "shadow_run_required must be false — the reuse/shadow-run branch "
            "is intentionally NOT implemented for V5.4.2",
        )
    role = str(a0.get("v541_e0_role") or "")
    if "parity" not in role or "historical" not in role:
        _err(
            "V542_A0_REUSE",
            "v541_e0_role must keep E0×3 as parity/historical reference only",
        )
    runs = a0.get("runs")
    expected = ["V542_A0_S17", "V542_A0_S29", "V542_A0_S43"]
    if not isinstance(runs, Sequence) or list(runs) != expected:
        _err(
            "V542_A0_REUSE",
            f"a0_baseline.runs must be exactly the matched retrain set {expected}",
        )
    # v1.1: template != instance; the INSTANCE is a T1 precondition.
    if a0.get("decision_record_template") != A0_DECISION_TEMPLATE:
        _err(
            "V542_A0_REUSE",
            "a0_baseline.decision_record_template must point at "
            f"{A0_DECISION_TEMPLATE!r}, got "
            f"{a0.get('decision_record_template')!r}",
        )
    if a0.get("decision_record_instance") != A0_DECISION_INSTANCE:
        _err(
            "V542_A0_REUSE",
            "a0_baseline.decision_record_instance must point at the planned "
            f"INSTANCE path {A0_DECISION_INSTANCE!r} (the template is NOT the "
            f"decision record), got {a0.get('decision_record_instance')!r}",
        )
    if a0.get("decision_record_instance_required") is not True:
        _err(
            "V542_A0_REUSE",
            "a0_baseline.decision_record_instance_required must be true — the "
            "a0_reuse_decision.json INSTANCE (decision=retrain_a0_x3) must "
            "exist before any formal T1 run",
        )


def _check_authorization(cfg: Mapping[str, Any]) -> None:
    gov = cfg.get("governance")
    auth = gov.get("authorization") if isinstance(gov, Mapping) else None
    if not isinstance(auth, Mapping):
        _err("V542_MISSING_FIELD", "governance.authorization missing")
    for flag in ("confirm_authorized", "sealed_authorized"):
        if auth.get(flag) is not False:
            _err(
                "V542_CONFIRM",
                f"authorization.{flag} must be false, got {auth.get(flag)!r}",
            )
    qc = auth.get("new_qc_calls")
    if qc != 0:
        _err(
            "V542_CONFIRM",
            f"authorization.new_qc_calls must be 0, got {qc!r}",
        )
    if auth.get("single_lane") is not True:
        _err(
            "V542_RESOURCE",
            f"authorization.single_lane must be true, got {auth.get('single_lane')!r}",
        )
    if auth.get("swap_allowed") is not False:
        _err(
            "V542_RESOURCE",
            f"authorization.swap_allowed must be false, got {auth.get('swap_allowed')!r}",
        )
    budget = cfg.get("budget")
    if isinstance(budget, Mapping):
        if budget.get("single_lane") is not True:
            _err("V542_RESOURCE", "budget.single_lane must be true")
        if budget.get("swap_max") != 0:
            _err(
                "V542_RESOURCE",
                f"budget.swap_max must be 0, got {budget.get('swap_max')!r}",
            )
        if budget.get("oom_kill_max") != 0:
            _err(
                "V542_RESOURCE",
                f"budget.oom_kill_max must be 0, got {budget.get('oom_kill_max')!r}",
            )


def _check_sampler_binding(cfg: Mapping[str, Any]) -> None:
    """V5.4.2 reuses the frozen V541 common sampler — config must mirror it."""
    tc = cfg.get("training_contract")
    binding = tc.get("sampler_binding") if isinstance(tc, Mapping) else None
    if not isinstance(binding, Mapping):
        _err(
            "V542_MISSING_FIELD", "training_contract.sampler_binding missing")
    try:
        from pes2ts_core.pesn.offline.training import v541_strata as strata
    except Exception as exc:  # noqa: BLE001 - fail-closed on ANY import error
        _err(
            "V542_SAMPLER_DRIFT",
            "cannot import the live v541_strata spec; refusing to validate a "
            f"sampler binding against code it cannot see ({exc})",
        )

    def _drift(what: str, got: Any, want: Any) -> None:
        _err(
            "V542_SAMPLER_DRIFT",
            f"sampler_binding.{what} = {got!r} != live code {want!r}",
        )

    if binding.get("sampler_id") != strata.SAMPLER_ID_V541:
        _drift("sampler_id", binding.get("sampler_id"), strata.SAMPLER_ID_V541)
    if binding.get("spec_sha256") != strata.sampler_spec_sha256():
        _drift("spec_sha256", binding.get("spec_sha256"), strata.sampler_spec_sha256())
    if tuple(binding.get("strata") or ()) != strata.STRATA_ORDER_V541:
        _drift("strata", binding.get("strata"), list(strata.STRATA_ORDER_V541))
    if binding.get("batch_curves") != strata.BATCH_CURVES_V541:
        _drift("batch_curves", binding.get("batch_curves"), strata.BATCH_CURVES_V541)
    if binding.get("frames_per_curve") != strata.FRAMES_PER_CURVE_V541:
        _drift("frames_per_curve", binding.get("frames_per_curve"),
               strata.FRAMES_PER_CURVE_V541)
    if dict(binding.get("mirror") or {}) != dict(strata.MIRROR_STRATA_V541):
        _drift("mirror", dict(binding.get("mirror") or {}),
               dict(strata.MIRROR_STRATA_V541))
    refs = binding.get("references")
    if not isinstance(refs, Mapping) or not str(
            refs.get("spec", "")).startswith(
        "pes2ts_core/pesn/offline/training/v541_strata.py#"):
        _err(
            "V542_SAMPLER_DRIFT",
            "sampler_binding.references.spec must point at "
            "v541_strata.py#SAMPLER_SPEC_V541",
        )


def _check_dose_blocks(cfg: Mapping[str, Any]) -> None:
    """Smoke / profile / diagnostic blocks (plan §7): fixed shapes."""
    smoke = cfg.get("smoke")
    if not isinstance(smoke, Mapping):
        _err("V542_MISSING_FIELD", "smoke must be a mapping")
    if smoke.get("updates") != 200:
        _err(
            "V542_RUN_MATRIX", f"smoke.updates must be 200, got {smoke.get('updates')!r}")
    if tuple(smoke.get("backbones") or ()) != ("A0", "A1", "A2", "A3", "A4", "A5"):
        _err(
            "V542_RUN_MATRIX", "smoke.backbones must be exactly A0–A5")
    if smoke.get("smoke_seed") != 17:
        _err(
            "V542_RUN_MATRIX",
            f"smoke.smoke_seed must be 17, got {smoke.get('smoke_seed')!r}",
        )
    if smoke.get("rank_excluded") is not True:
        _err(
            "V542_RUN_MATRIX", "smoke must be rank_excluded")

    profile = cfg.get("profile")
    if not isinstance(profile, Mapping):
        _err("V542_MISSING_FIELD", "profile must be a mapping")
    window = profile.get("window")
    if not isinstance(window, Mapping) or (
            window.get("w0_updates"), window.get("w1_updates")) != (40, 40):
        _err(
            "V542_RUN_MATRIX", "profile window must be 40 W0 + 40 W1")
    if "V542" not in str(profile.get("denominator") or ""):
        _err(
            "V542_RUN_MATRIX",
            "profile.denominator must be the V542 A0 same-env measurement "
            "(V541 archived cost table is forbidden)",
        )
    forbidden_den = str(profile.get("forbidden_denominator") or "")
    if "v541" not in forbidden_den.lower():
        _err(
            "V542_RUN_MATRIX",
            "profile must explicitly forbid the V541 archived denominator",
        )
    if tuple(profile.get("conditional_backbones") or ()) != ("A1-C", "A5-C"):
        _err(
            "V542_RUN_MATRIX",
            "profile.conditional_backbones must be exactly ['A1-C', 'A5-C'] "
            "(the full-dose attribution diagnostics need their own 40+40 "
            f"profile + cost gate before activation), got "
            f"{profile.get('conditional_backbones')!r}",
        )

    diag = cfg.get("diagnostic")
    if not isinstance(diag, Mapping):
        _err("V542_MISSING_FIELD", "diagnostic must be a mapping")
    if diag.get("updates") != 2000 or diag.get("w0_updates") != 400 \
            or diag.get("w1_updates") != 1600:
        _err(
            "V542_RUN_MATRIX",
            "diagnostic dose must be 2000 = 400 W0 + 1600 W1",
        )
    if diag.get("seed") != 17:
        _err(
            "V542_RUN_MATRIX",
            f"diagnostic.seed must be 17, got {diag.get('seed')!r}",
        )
    if diag.get("a0_reference") != "V542_DIAG_A0_S17":
        _err(
            "V542_RUN_MATRIX",
            "diagnostic.a0_reference must be V542_DIAG_A0_S17 (matched-dose "
            "reference; without it the screen systematically eliminates all)",
        )
    if diag.get("elimination_only") is not True or \
            diag.get("forbids_full_dose_h3_effect_gates") is not True:
        _err(
            "V542_RUN_MATRIX",
            "diagnostic must be elimination-only and must forbid applying the "
            "full-dose H3 effect gates at 2,000 updates",
        )
    if diag.get("rank_excluded") is not True:
        _err(
            "V542_RUN_MATRIX", "diagnostic must be rank_excluded")


def _check_selection_tau(cfg: Mapping[str, Any]) -> None:
    """Checkpoint selection and tau calibration stay inner-val only (plan §9)."""
    tc = cfg.get("training_contract")
    selection = tc.get("selection") if isinstance(tc, Mapping) else None
    if not isinstance(selection, Mapping) or \
            selection.get("metric") != "inner-val total loss":
        _err(
            "V542_RUN_MATRIX",
            "training_contract.selection.metric must be 'inner-val total loss' "
            "(checkpoints are never selected by H3 metrics)",
        )
    tau = tc.get("tau_end") if isinstance(tc, Mapping) else None
    if not isinstance(tau, Mapping) or tau.get("calibrated_on") != "inner-val":
        _err(
            "V542_RUN_MATRIX",
            "tau_end.calibrated_on must be 'inner-val' (never dev)",
        )


def _check_top2_and_a5_supplement(cfg: Mapping[str, Any], runs: Sequence[Any]) -> None:
    """v1.1 top-2 / A5-supplement split (plan §8.1; Oracle findings 1–2).

    top2 = A1–A4 × {29,43} = 8 preregistered rows, max 4 activated; A5 is
    NEVER in top2 — its T3 rows are a separate a5_supplement branch (max 2)
    gated by ``a5_full_dose_complete_and_seed17_screen_pass`` (a seed17
    SCREEN predicate, not a three-seed H3 claim).
    """
    rm = cfg.get("run_matrix")
    block = rm.get("top2_preregistration")
    if not isinstance(block, Mapping):
        _err(
            "V542_RUN_MATRIX", "run_matrix.top2_preregistration block missing"
        )
    candidates = tuple(block.get("candidates") or ())
    if "A5" in candidates:
        _err(
            "V542_RUN_MATRIX",
            "top2_preregistration.candidates must NOT include A5 — A5's T3 "
            "rows belong to the separate a5_supplement branch "
            f"(parent gate {A5_SUPPLEMENT_PARENT_GATE!r}), got "
            f"{list(candidates)!r}",
        )
    if candidates != TOP2_CANDIDATES:
        _err(
            "V542_RUN_MATRIX",
            f"top2_preregistration.candidates must be {list(TOP2_CANDIDATES)}, "
            f"got {block.get('candidates')!r}",
        )
    if tuple(block.get("seeds") or ()) != T3_SEEDS:
        _err(
            "V542_RUN_MATRIX",
            f"top2_preregistration.seeds must be {list(T3_SEEDS)}, "
            f"got {block.get('seeds')!r}",
        )
    if block.get("listed_runs") != TOP2_LISTED_RUNS:
        _err(
            "V542_RUN_MATRIX",
            f"top2_preregistration.listed_runs must be {TOP2_LISTED_RUNS} "
            "(A1-A4 x seeds 29/43)",
        )
    if block.get("top2_max_runs") != TOP2_MAX_RUNS:
        _err(
            "V542_RUN_MATRIX",
            f"top2_preregistration.top2_max_runs must be {TOP2_MAX_RUNS} "
            "(<=2 activated candidates x 2 seeds; the superseded "
            "full_activation_count contract is gone)",
        )
    if block.get("full_activation_count") is not None:
        _err(
            "V542_RUN_MATRIX",
            "top2_preregistration.full_activation_count is superseded by "
            "top2_max_runs/a5_supplement_max_runs/t3_max_runs and must be "
            "removed",
        )
    if block.get("a5_supplement_max_runs") != A5_SUPPLEMENT_MAX_RUNS:
        _err(
            "V542_RUN_MATRIX",
            "top2_preregistration.a5_supplement_max_runs must be "
            f"{A5_SUPPLEMENT_MAX_RUNS} (A5 T3 supplement counted separately "
            f"from top2), got {block.get('a5_supplement_max_runs')!r}",
        )
    if block.get("t3_max_runs") != T3_MAX_RUNS:
        _err(
            "V542_RUN_MATRIX",
            "top2_preregistration.t3_max_runs must be "
            f"{T3_MAX_RUNS} (top2 <=4 + a5_supplement <=2), got "
            f"{block.get('t3_max_runs')!r}",
        )
    if block.get("activation") != T3_ACTIVATION:
        _err(
            "V542_RUN_MATRIX",
            f"top2_preregistration.activation must be {T3_ACTIVATION!r}",
        )
    if block.get("unactivated_status") != T3_UNACTIVATED_STATUS:
        _err(
            "V542_RUN_MATRIX",
            "top2_preregistration.unactivated_status must be "
            f"{T3_UNACTIVATED_STATUS!r}",
        )
    if block.get("n_core_counting") != "formula_only":
        _err(
            "V542_RUN_MATRIX",
            "top2_preregistration.n_core_counting must be 'formula_only'",
        )
    if block.get("seed17_screen_pass_ref") != SCREEN_PREDICATE_REF:
        _err(
            "V542_RUN_MATRIX",
            f"top2_preregistration.seed17_screen_pass_ref must be "
            f"{SCREEN_PREDICATE_REF!r}, got "
            f"{block.get('seed17_screen_pass_ref')!r}",
        )
    if "A0_S17" not in str(block.get("freeze_requires") or ""):
        _err(
            "V542_RUN_MATRIX",
            "top2_preregistration.freeze_requires must pin the A0_S17 "
            "completion dependency (H2/top2 need the matched comparator), got "
            f"{block.get('freeze_requires')!r}",
        )
    if TOP2_MAX_RUNS + A5_SUPPLEMENT_MAX_RUNS != T3_MAX_RUNS:
        _err(  # pragma: no cover - defensive: constants are frozen
            "V542_RUN_MATRIX",
            "internal invariant broken: top2_max_runs + a5_supplement_max_runs "
            "must equal t3_max_runs",
        )

    # top-2 listed rows: exactly the 8 A1-A4 x {29,43} runs, A5 excluded.
    top2_rows = [
        r for r in runs
        if isinstance(r, Mapping) and r.get("condition") == T3_ACTIVATION
    ]
    expected_top2_ids = {
        f"V542_{bid}_S{seed}" for bid in TOP2_CANDIDATES for seed in T3_SEEDS
    }
    actual_top2_ids = {r.get("run_id") for r in top2_rows}
    if actual_top2_ids != expected_top2_ids:
        missing = sorted(expected_top2_ids - actual_top2_ids)
        extra = sorted(x for x in actual_top2_ids - expected_top2_ids if x)
        _err(
            "V542_RUN_MATRIX",
            f"top-2 preregistration must list exactly the 8 A1-A4 x seeds "
            f"{list(T3_SEEDS)} runs (A5 never enters top2); "
            f"missing={missing} extra={extra}",
        )
    if len(top2_rows) != TOP2_LISTED_RUNS:
        _err(
            "V542_RUN_MATRIX", "duplicate/overlapping top-2 run entries detected"
        )
    for run in top2_rows:
        run_id = str(run.get("run_id"))
        if T3_RUN_ID_TOKEN in run_id:
            _err(
                "V542_RUN_MATRIX",
                f"{run_id}: ambiguous TOP2 placeholder ids are forbidden; every "
                "T3 run carries a deterministic backbone identity",
            )
        if run.get("class") != "conditional":
            _err(
                "V542_RUN_MATRIX", f"{run_id}: T3 runs must be class=conditional"
            )
        if run.get("winner_eligible") is not True:
            _err(
                "V542_RUN_MATRIX", f"{run_id}: T3 runs must be winner_eligible=true"
            )
        if run.get("unactivated_status") != T3_UNACTIVATED_STATUS:
            _err(
                "V542_RUN_MATRIX",
                f"{run_id}: unactivated_status must be {T3_UNACTIVATED_STATUS!r}",
            )
        if run.get("new_training_if_activated") != 1:
            _err(
                "V542_RUN_MATRIX",
                f"{run_id}: must declare new_training_if_activated=1",
            )
        if run.get("new_training") is not None:
            _err(
                "V542_RUN_MATRIX",
                f"{run_id}: T3 entries must not carry unconditional 'new_training'",
            )
        if run.get("backbone") is None:
            _err(
                "V542_BACKBONE_UNKNOWN",
                f"{run_id}: run lacks a deterministic backbone identity",
            )
        if "parent_gate" in run:
            _err(
                "V542_RUN_MATRIX",
                f"{run_id}: parent_gate is only valid on A5 supplement T3 rows",
            )

    # A5 supplement branch — fully separate from top2.
    sup = rm.get("a5_supplement")
    if not isinstance(sup, Mapping):
        _err(
            "V542_RUN_MATRIX",
            "run_matrix.a5_supplement block missing — A5's T3 rows are not "
            "part of top2 and must carry their own supplement branch",
        )
    if tuple(sup.get("runs") or ()) != ("V542_A5_S29", "V542_A5_S43"):
        _err(
            "V542_RUN_MATRIX",
            "a5_supplement.runs must be exactly [V542_A5_S29, V542_A5_S43], "
            f"got {sup.get('runs')!r}",
        )
    if sup.get("a5_supplement_max_runs") != A5_SUPPLEMENT_MAX_RUNS:
        _err(
            "V542_RUN_MATRIX",
            f"a5_supplement.a5_supplement_max_runs must be "
            f"{A5_SUPPLEMENT_MAX_RUNS}",
        )
    if sup.get("condition") != A5_SUPPLEMENT_CONDITION or \
            sup.get("parent_gate") != A5_SUPPLEMENT_PARENT_GATE:
        _err(
            "V542_RUN_MATRIX",
            "a5_supplement condition/parent_gate must both be "
            f"{A5_SUPPLEMENT_PARENT_GATE!r} (A5 full-dose complete AND A5 "
            "seed17_screen_pass — a seed17 SCREEN, never a three-seed H3 "
            f"claim); got condition={sup.get('condition')!r} "
            f"parent_gate={sup.get('parent_gate')!r}",
        )
    if sup.get("screen_predicate_ref") != SCREEN_PREDICATE_REF:
        _err(
            "V542_RUN_MATRIX",
            f"a5_supplement.screen_predicate_ref must be "
            f"{SCREEN_PREDICATE_REF!r}",
        )
    a5_t3 = [
        r for r in runs
        if isinstance(r, Mapping) and r.get("stage") == "T3"
        and r.get("backbone") == "A5"
    ]
    a5_t3_ids = {r.get("run_id") for r in a5_t3}
    if a5_t3_ids != {"V542_A5_S29", "V542_A5_S43"} or len(a5_t3) != 2:
        _err(
            "V542_RUN_MATRIX",
            "A5 T3 rows must be exactly [V542_A5_S29, V542_A5_S43] under the "
            f"a5_supplement branch, got {sorted(a5_t3_ids)}",
        )
    for run in a5_t3:
        run_id = str(run.get("run_id"))
        if run.get("condition") != A5_SUPPLEMENT_CONDITION or \
                run.get("parent_gate") != A5_SUPPLEMENT_PARENT_GATE:
            _err(
                "V542_RUN_MATRIX",
                f"{run_id}: A5 supplement run must stack parent_gate="
                f"{A5_SUPPLEMENT_PARENT_GATE!r} with matching condition "
                f"(old 'a5_full_dose_activated_and_seed17_h3_pass' is "
                f"superseded); got condition={run.get('condition')!r} "
                f"parent_gate={run.get('parent_gate')!r}",
            )

    t3_total = [r for r in runs if isinstance(r, Mapping) and r.get("stage") == "T3"]
    if len(t3_total) != TOP2_LISTED_RUNS + A5_SUPPLEMENT_MAX_RUNS:
        _err(
            "V542_RUN_MATRIX",
            f"T3 must list exactly {TOP2_LISTED_RUNS} top-2 rows + "
            f"{A5_SUPPLEMENT_MAX_RUNS} A5 supplement rows = "
            f"{TOP2_LISTED_RUNS + A5_SUPPLEMENT_MAX_RUNS} preregistered rows "
            f"(activation ceiling t3_max_runs={T3_MAX_RUNS}); got "
            f"{len(t3_total)}",
        )


def _check_activation_branches(cfg: Mapping[str, Any], runs: Sequence[Any]) -> None:
    """A5 full-dose / A1-C / A5-C preregistered activation rules (plan §8.1)."""
    by_id = {r.get("run_id"): r for r in runs if isinstance(r, Mapping)}

    a5 = by_id.get("V542_A5_S17")
    if a5 is None:
        _err(
            "V542_RUN_MATRIX", "V542_A5_S17 conditional run must be preregistered")
    if a5.get("class") != "conditional" or a5.get("winner_eligible") is not True:
        _err(
            "V542_RUN_MATRIX",
            "V542_A5_S17 must be class=conditional and winner_eligible=true",
        )
    if a5.get("condition") != A5_FULLDOSE_CONDITION:
        _err(
            "V542_RUN_MATRIX",
            f"V542_A5_S17.condition must be {A5_FULLDOSE_CONDITION!r} "
            f"(auto-activate ONLY when no A1–A4 H3-qualified candidate exists), "
            f"got {a5.get('condition')!r}",
        )
    if a5.get("unactivated_status") != T3_UNACTIVATED_STATUS or \
            a5.get("new_training_if_activated") != 1:
        _err(
            "V542_RUN_MATRIX",
            "V542_A5_S17 must carry unactivated_status=gate_skipped and "
            "new_training_if_activated=1",
        )
    if tuple(a5.get("launch_requires") or ()) != A5_FULLDOSE_LAUNCH_REQUIRES:
        _err(
            "V542_RUN_MATRIX",
            "V542_A5_S17.launch_requires must be exactly "
            f"{list(A5_FULLDOSE_LAUNCH_REQUIRES)} (diagnostic survival + cost "
            "+ legality + resource before full dose), got "
            f"{a5.get('launch_requires')!r}",
        )

    a1c = by_id.get("V542_A1C_S17")
    if a1c is None:
        _err(
            "V542_RUN_MATRIX", "V542_A1C_S17 attribution run must be preregistered")
    if a1c.get("class") != "diagnostic" or a1c.get("winner_eligible") is not False:
        _err(
            "V542_RUN_MATRIX",
            "V542_A1C_S17 must be class=diagnostic and never winner-eligible",
        )
    if a1c.get("condition") != A1C_ACTIVATION:
        _err(
            "V542_RUN_MATRIX",
            f"V542_A1C_S17.condition must be {A1C_ACTIVATION!r} (activation on "
            f"entering the top-2 set OR selection — not top-ranked only), "
            f"got {a1c.get('condition')!r}",
        )
    if a1c.get("skip_reason") != A1C_SKIP_REASON:
        _err(
            "V542_RUN_MATRIX",
            f"V542_A1C_S17.skip_reason must be {A1C_SKIP_REASON!r}, got "
            f"{a1c.get('skip_reason')!r}",
        )

    a5c = by_id.get("V542_A5C_S17")
    if a5c is None:
        _err(
            "V542_RUN_MATRIX", "V542_A5C_S17 attribution run must be preregistered")
    if a5c.get("class") != "diagnostic" or a5c.get("winner_eligible") is not False:
        _err(
            "V542_RUN_MATRIX",
            "V542_A5C_S17 must be class=diagnostic and never winner-eligible",
        )
    if a5c.get("condition") != A5C_ACTIVATION:
        _err(
            "V542_RUN_MATRIX",
            f"V542_A5C_S17.condition must be {A5C_ACTIVATION!r} (A5-C runs "
            f"ONLY when A5 is SELECTED as Backbone*; the superseded "
            f"'a5_promoted_after_h3' predicate is forbidden), got "
            f"{a5c.get('condition')!r}",
        )
    if a5c.get("skip_reason") != A5C_SKIP_REASON:
        _err(
            "V542_RUN_MATRIX",
            f"V542_A5C_S17.skip_reason must be {A5C_SKIP_REASON!r}, got "
            f"{a5c.get('skip_reason')!r}",
        )

    # A5 diagnostic must be unconditional (smoke/profile/diagnostic first,
    # no discretion) — only the full-dose branch is conditional.
    a5_diag = by_id.get("V542_DIAG_A5_S17")
    if a5_diag is None:
        _err(
            "V542_RUN_MATRIX",
            "V542_DIAG_A5_S17 must be preregistered (A5 smoke/profile/"
            "diagnostic always complete before any activation decision)",
        )
    if a5_diag.get("class") != "diagnostic" or \
            a5_diag.get("new_training") != 1 or \
            a5_diag.get("condition") is not None:
        _err(
            "V542_RUN_MATRIX",
            "V542_DIAG_A5_S17 must be an unconditional diagnostic run "
            "(new_training=1, no condition) — only A5 full-dose is conditional",
        )

    # the A5 activation / A1-C / A5-C rules must also live in the run_matrix prose
    rm = cfg.get("run_matrix")
    a5_rules = rm.get("a5_activation")
    if not isinstance(a5_rules, Mapping) or \
            a5_rules.get("no_discretion") is not True:
        _err(
            "V542_RUN_MATRIX",
            "run_matrix.a5_activation must declare no_discretion=true",
        )
    if isinstance(a5_rules, Mapping):
        if tuple(a5_rules.get("launch_requires") or ()) != \
                A5_FULLDOSE_LAUNCH_REQUIRES:
            _err(
                "V542_RUN_MATRIX",
                "run_matrix.a5_activation.launch_requires must be exactly "
                f"{list(A5_FULLDOSE_LAUNCH_REQUIRES)} — no A1-A4 final-H3 "
                f"qualification may sit in A5's launch gates, got "
                f"{a5_rules.get('launch_requires')!r}",
            )
        if a5_rules.get("full_dose_condition") != A5_FULLDOSE_CONDITION:
            _err(
                "V542_RUN_MATRIX",
                "run_matrix.a5_activation.full_dose_condition must be "
                f"{A5_FULLDOSE_CONDITION!r}, got "
                f"{a5_rules.get('full_dose_condition')!r}",
            )
        if a5_rules.get("diagnostic_survival_required") is not True:
            _err(
                "V542_RUN_MATRIX",
                "run_matrix.a5_activation.diagnostic_survival_required must be "
                "true (A5 must not auto-activate after failing its own "
                "diagnostic)",
            )
        if a5_rules.get("smoke_profile_diagnostic_first") is not True:
            _err(
                "V542_RUN_MATRIX",
                "run_matrix.a5_activation.smoke_profile_diagnostic_first must "
                "be true",
            )
    a1c_rules = rm.get("a1c_activation")
    if not isinstance(a1c_rules, Mapping) or \
            a1c_rules.get("winner_eligible") is not False:
        _err(
            "V542_RUN_MATRIX",
            "run_matrix.a1c_activation must keep winner_eligible=false",
        )
    if isinstance(a1c_rules, Mapping):
        if a1c_rules.get("not_only_top_ranked") is not True:
            _err(
                "V542_RUN_MATRIX",
                "run_matrix.a1c_activation.not_only_top_ranked must be true "
                "(frozen spec override: top-2 set OR selection, not top-ranked only)",
            )
        if a1c_rules.get("condition") != A1C_ACTIVATION or \
                a1c_rules.get("skip_reason") != A1C_SKIP_REASON:
            _err(
                "V542_RUN_MATRIX",
                "run_matrix.a1c_activation must restate condition="
                f"{A1C_ACTIVATION!r} and skip_reason={A1C_SKIP_REASON!r}",
            )
        if tuple(a1c_rules.get("activation_requires") or ()) != \
                ATTR_ACTIVATION_REQUIRES:
            _err(
                "V542_RUN_MATRIX",
                "run_matrix.a1c_activation.activation_requires must be exactly "
                f"{list(ATTR_ACTIVATION_REQUIRES)} (incl. budget_record), got "
                f"{a1c_rules.get('activation_requires')!r}",
            )
    a5c_rules = rm.get("a5c_activation")
    if not isinstance(a5c_rules, Mapping):
        _err(
            "V542_RUN_MATRIX", "run_matrix.a5c_activation block missing")
    else:
        if a5c_rules.get("condition") != A5C_ACTIVATION:
            _err(
                "V542_RUN_MATRIX",
                "run_matrix.a5c_activation.condition must be "
                f"{A5C_ACTIVATION!r} (A5 SELECTED as Backbone* only), got "
                f"{a5c_rules.get('condition')!r}",
            )
        if a5c_rules.get("winner_eligible") is not False:
            _err(
                "V542_RUN_MATRIX",
                "run_matrix.a5c_activation must keep winner_eligible=false",
            )
        if a5c_rules.get("skip_reason") != A5C_SKIP_REASON:
            _err(
                "V542_RUN_MATRIX",
                "run_matrix.a5c_activation.skip_reason must be "
                f"{A5C_SKIP_REASON!r}",
            )
        if tuple(a5c_rules.get("activation_requires") or ()) != \
                ATTR_ACTIVATION_REQUIRES:
            _err(
                "V542_RUN_MATRIX",
                "run_matrix.a5c_activation.activation_requires must be exactly "
                f"{list(ATTR_ACTIVATION_REQUIRES)} (incl. budget_record), got "
                f"{a5c_rules.get('activation_requires')!r}",
            )
        if "capacity" not in str(a5c_rules.get("capacity_caveat_rule") or ""):
            _err(
                "V542_RUN_MATRIX",
                "run_matrix.a5c_activation.capacity_caveat_rule must record "
                "capacity_caveat_unresolved when A5 is H3-qualified but not "
                "selected",
            )


def _check_run_matrix(cfg: Mapping[str, Any]) -> None:
    rm = cfg.get("run_matrix")
    if not isinstance(rm, Mapping):
        _err("V542_MISSING_FIELD", "run_matrix must be a mapping")
    classes = rm.get("classes")
    if not isinstance(classes, Sequence) or set(classes) != RUN_CLASSES:
        _err(
            "V542_RUN_MATRIX",
            f"run_matrix.classes must be exactly {sorted(RUN_CLASSES)}, "
            f"got {classes!r}",
        )
    runs = rm.get("runs")
    if not isinstance(runs, Sequence) or not runs:
        _err("V542_MISSING_FIELD", "run_matrix.runs missing/empty")
    seeds = set(cfg.get("split_contract", {}).get("seeds") or [])
    seen_ids: set[str] = set()
    winners = 0
    a0_ids: set[str] = set()
    for entry in runs:
        if not isinstance(entry, Mapping):
            _err("V542_RUN_MATRIX", f"run entry not a mapping: {entry!r}")
        run_id = entry.get("run_id")
        if not isinstance(run_id, str) or not run_id.startswith("V542_"):
            _err(
                "V542_RUN_MATRIX",
                f"run_id {run_id!r} must use the V542_ namespace",
            )
        if T3_RUN_ID_TOKEN in run_id:
            _err(
                "V542_RUN_MATRIX",
                f"{run_id}: ambiguous TOP2 placeholder ids are forbidden",
            )
        if run_id in seen_ids:
            _err("V542_RUN_MATRIX", f"duplicate run_id {run_id!r}")
        seen_ids.add(run_id)
        cls = entry.get("class")
        if cls not in RUN_CLASSES:
            _err(
                "V542_RUN_MATRIX",
                f"{run_id}: class {cls!r} not in {sorted(RUN_CLASSES)}",
            )
        if seeds and entry.get("seed") not in seeds:
            _err(
                "V542_RUN_MATRIX",
                f"{run_id}: seed {entry.get('seed')!r} not in split seeds "
                f"{sorted(seeds)}",
            )
        bid = entry.get("backbone")
        if bid is None:
            _err(
                "V542_BACKBONE_UNKNOWN",
                f"{run_id}: run lacks a deterministic backbone identity",
            )
        if bid not in KNOWN_BACKBONES:
            _err(
                "V542_BACKBONE_UNKNOWN",
                f"{run_id}: backbone {bid!r} outside frozen registry",
            )
        if bid in DIAGNOSTIC_BACKBONES and cls != "diagnostic":
            _err(
                "V542_RUN_MATRIX",
                f"{run_id}: diagnostic-only backbone {bid!r} may not occupy a "
                f"{cls!r} run",
            )
        if bool(entry.get("winner_eligible")):
            if cls == "diagnostic":
                _err(
                    "V542_RUN_MATRIX",
                    f"{run_id}: diagnostic runs are never winner-eligible",
                )
            if bid not in WINNER_BACKBONES:
                _err(
                    "V542_RUN_MATRIX",
                    f"{run_id}: only A1–A5 may be winner-eligible; got {bid!r} "
                    "(A0 is the matched baseline)",
                )
            winners += 1
        if bid == "A0":
            if bool(entry.get("winner_eligible")):
                _err(
                    "V542_RUN_MATRIX",
                    f"{run_id}: A0 matched baseline must never be winner-eligible",
                )
            if entry.get("dose") == "formal":
                a0_ids.add(run_id)
                if entry.get("role") != "matched_baseline_retrain":
                    _err(
                        "V542_RUN_MATRIX",
                        f"{run_id}: formal-dose A0 runs must declare "
                        "role=matched_baseline_retrain (retrain_a0_x3 decision)",
                    )
            elif entry.get("dose") == "diagnostic_2000":
                if entry.get("role") != "matched_dose_reference":
                    _err(
                        "V542_RUN_MATRIX",
                        f"{run_id}: diagnostic A0@2000 must declare "
                        "role=matched_dose_reference",
                    )
    if a0_ids != {"V542_A0_S17", "V542_A0_S29", "V542_A0_S43"}:
        _err(
            "V542_RUN_MATRIX",
            f"matched A0 retrain must register exactly seeds 17/29/43, got "
            f"{sorted(a0_ids)}",
        )
    if winners == 0:
        _err(
            "V542_RUN_MATRIX", "no winner-eligible (nominal/conditional) run registered"
        )

    # v1.1 hardening (Oracle finding 9): the run set is EXACT — 26 rows with
    # frozen stage/class/dose/backbone/seed/new-training/winner mapping.
    actual_ids = {str(r.get("run_id")) for r in runs if isinstance(r, Mapping)}
    if actual_ids != EXPECTED_RUN_IDS or len(runs) != EXPECTED_RUN_COUNT:
        missing = sorted(EXPECTED_RUN_IDS - actual_ids)
        extra = sorted(actual_ids - EXPECTED_RUN_IDS)
        _err(
            "V542_RUN_MATRIX",
            f"run_matrix.runs must be exactly the frozen {EXPECTED_RUN_COUNT}-run "
            f"matrix (A0x3 + A1-A4 seed17 + A5_S17 + 8 top-2 rows + 2 A5 "
            f"supplement rows + 6 diagnostics + A1C/A5C); "
            f"missing={missing} extra={extra} count={len(runs)}",
        )
    for entry in runs:
        run_id = str(entry.get("run_id"))
        expected = EXPECTED_RUN_TABLE[run_id]
        for field, want in expected.items():
            got = entry.get(field)
            if isinstance(want, list):
                if list(got or ()) != want:
                    _err(
                        "V542_RUN_MATRIX",
                        f"{run_id}.{field} must be {want!r}, got {got!r}",
                    )
            elif got != want:
                _err(
                    "V542_RUN_MATRIX",
                    f"{run_id}.{field} must be {want!r}, got {got!r}",
                )
        if "new_training_if_activated" in expected:
            if entry.get("new_training") is not None:
                _err(
                    "V542_RUN_MATRIX",
                    f"{run_id}: conditional rows must not carry unconditional "
                    "'new_training'",
                )
        elif entry.get("new_training_if_activated") is not None:
            _err(
                "V542_RUN_MATRIX",
                f"{run_id}: unconditional rows must not carry "
                "'new_training_if_activated'",
            )


def _check_gv_block(cfg: Mapping[str, Any]) -> None:
    gv = cfg.get("gv")
    if not isinstance(gv, Mapping):
        _err("V542_MISSING_FIELD", "gv must be a mapping")
    missing = [k for k in _GV_DEFAULT if gv.get(k) is None]
    if missing:
        _err("V542_MISSING_FIELD", f"gv missing field(s): {missing}")
    try:
        gv_map = load_gv_map(DEFAULT_MAP_PATH)
        campaign = resolve_campaign(gv_map, str(gv["campaign_id"]))
    except GVMapError as exc:
        _err("V542_GV_MISMATCH", str(exc))
    if gv["campaign_id"] != GV_CAMPAIGN_ID:
        _err(
            "V542_GV_MISMATCH",
            f"gv.campaign_id must be {GV_CAMPAIGN_ID!r}, got {gv['campaign_id']!r}",
        )
    if gv.get("primary_g") != campaign.primary_g:
        _err(
            "V542_GV_MISMATCH",
            f"gv.primary_g {gv.get('primary_g')!r} != registry {campaign.primary_g!r}",
        )
    if set(gv.get("supporting_g") or ()) != set(campaign.supporting_g):
        _err(
            "V542_GV_MISMATCH",
            f"gv.supporting_g {sorted(gv.get('supporting_g') or ())} != registry "
            f"{sorted(campaign.supporting_g)}",
        )
    if set(gv.get("gates") or ()) != set(campaign.gates):
        _err(
            "V542_GV_MISMATCH",
            f"gv.gates {sorted(gv.get('gates') or ())} != registry "
            f"{sorted(campaign.gates)}",
        )
    if gv.get("map_version") != gv_map.map_version:
        _err(
            "V542_GV_MISMATCH",
            f"gv.map_version {gv.get('map_version')!r} != {gv_map.map_version!r}",
        )
    if campaign.status not in {"proposed", "active"}:
        _err(
            "V542_GV_MISMATCH",
            f"registry {campaign.campaign_id} status {campaign.status!r} — WP0 "
            "registration must stay proposed/active, never executed",
        )


def _check_statistics(cfg: Mapping[str, Any]) -> None:
    """v1.1 frozen H3 statistics contract (Oracle finding 3).

    fix-6 C6: the structured fields themselves are asserted exactly —
    ``ci``/``pairing``/``final_draws_ref``/all four improvement directions/
    the exact p numerator/same CI-p draws/frame-level bans/Holm families
    — not merely the token presence.  Correction (gate-1 C5): the
    ``h3_eligible_candidates`` eligible set and ``champion_rule`` are
    TOKEN-presence checks, NOT exact assertions — converting them to
    exact frozen strings is open gate-0 advisory A2 (close before
    P3/WP6).
    """
    ec = cfg.get("evaluation_contract")
    stats = ec.get("statistics") if isinstance(ec, Mapping) else None
    if not isinstance(stats, Mapping):
        _err(
            "V542_MISSING_FIELD", "evaluation_contract.statistics missing")
    if stats.get("bootstrap_seed_root") != BOOTSTRAP_SEED_ROOT:
        _err(
            "V542_RUN_MATRIX",
            f"statistics.bootstrap_seed_root must be {BOOTSTRAP_SEED_ROOT}, got "
            f"{stats.get('bootstrap_seed_root')!r}",
        )
    if stats.get("bootstrap_subseed_formula") != BOOTSTRAP_SUBSEED_FORMULA:
        _err(
            "V542_RUN_MATRIX",
            "statistics.bootstrap_subseed_formula must be the frozen "
            f"deterministic sub-seed formula {BOOTSTRAP_SUBSEED_FORMULA!r}, got "
            f"{stats.get('bootstrap_subseed_formula')!r}",
        )
    if stats.get("bootstrap_unit") != "curve_reaction_cluster":
        _err(
            "V542_RUN_MATRIX",
            "statistics.bootstrap_unit must be curve_reaction_cluster "
            f"(curve-level clustering; frame-level is forbidden), got "
            f"{stats.get('bootstrap_unit')!r}",
        )
    if stats.get("initial_draws") != INITIAL_DRAWS or \
            stats.get("final_draws") != FINAL_DRAWS:
        _err(
            "V542_RUN_MATRIX",
            f"statistics draws must be initial={INITIAL_DRAWS} / "
            f"final={FINAL_DRAWS}, got {stats.get('initial_draws')!r}/"
            f"{stats.get('final_draws')!r}",
        )
    if stats.get("final_draws_ref") != FINAL_DRAWS_REF:
        _err(
            "V542_RUN_MATRIX",
            f"statistics.final_draws_ref must be {FINAL_DRAWS_REF!r} (the "
            "signed threshold is the single source of truth), got "
            f"{stats.get('final_draws_ref')!r}",
        )
    th = cfg.get("thresholds")
    if isinstance(th, Mapping) and th.get("final_bootstrap_draws") is not None \
            and stats.get("final_draws") != th.get("final_bootstrap_draws"):
        _err(
            "V542_RUN_MATRIX",
            "statistics.final_draws must equal "
            f"thresholds.final_bootstrap_draws ({th.get('final_bootstrap_draws')!r}), "
            f"got {stats.get('final_draws')!r}",
        )
    if stats.get("ci") != CI_METHOD:
        _err(
            "V542_RUN_MATRIX",
            f"statistics.ci must be {CI_METHOD!r} (frozen 95% percentile "
            f"interval), got {stats.get('ci')!r}",
        )
    if stats.get("pairing") != PAIRING_SENTENCE:
        _err(
            "V542_RUN_MATRIX",
            "statistics.pairing must be the frozen sentence "
            f"{PAIRING_SENTENCE!r}, got {stats.get('pairing')!r}",
        )
    if stats.get("ci_and_p_same_draws") is not True:
        _err(
            "V542_RUN_MATRIX",
            "statistics.ci_and_p_same_draws must be true (CI and p-value share "
            "one set of paired cluster bootstrap draws)",
        )
    if stats.get("frame_level_bootstrap_forbidden") is not True:
        _err(
            "V542_RUN_MATRIX",
            "statistics.frame_level_bootstrap_forbidden must be true",
        )
    p_rule = stats.get("p_value_rule")
    if not isinstance(p_rule, Mapping):
        _err(
            "V542_MISSING_FIELD", "statistics.p_value_rule missing")
    if p_rule.get("type") != P_VALUE_TYPE:
        _err(
            "V542_RUN_MATRIX",
            f"statistics.p_value_rule.type must be {P_VALUE_TYPE!r}, got "
            f"{p_rule.get('type')!r}",
        )
    if p_rule.get("formula") != P_FORMULA_EXACT:
        _err(
            "V542_RUN_MATRIX",
            "statistics.p_value_rule.formula must be the exact one-sided "
            f"improvement rule {P_FORMULA_EXACT!r} (numerator 1 + count, "
            f"denominator 10001), got {p_rule.get('formula')!r}",
        )
    if p_rule.get("draws") != FINAL_DRAWS:
        _err(
            "V542_RUN_MATRIX",
            f"statistics.p_value_rule.draws must be {FINAL_DRAWS} (same draws "
            f"as the CI), got {p_rule.get('draws')!r}",
        )
    if p_rule.get("frame_level_p_forbidden") is not True:
        _err(
            "V542_RUN_MATRIX",
            "statistics.p_value_rule.frame_level_p_forbidden must be true",
        )
    direction = p_rule.get("improvement_direction")
    if not isinstance(direction, Mapping) or \
            dict(direction) != IMPROVEMENT_DIRECTIONS:
        _err(
            "V542_RUN_MATRIX",
            "statistics.p_value_rule.improvement_direction must be exactly "
            f"{IMPROVEMENT_DIRECTIONS} (candidate_minus_a0 for direction/"
            "AUPRC/q, a0_minus_candidate for rmse_zero), got "
            f"{dict(direction) if isinstance(direction, Mapping) else direction!r}",
        )
    holm = stats.get("holm_family")
    if not isinstance(holm, Mapping):
        _err(
            "V542_MISSING_FIELD", "statistics.holm_family missing")
    for exit_name in ("move_direction_exit", "move_rmse_exit"):
        fam = holm.get(exit_name)
        if not isinstance(fam, Mapping) or float(fam.get("alpha") or 0) != 0.025 \
                or fam.get("correction") != "holm" or \
                fam.get("scope") != "eligible_candidate_set_within_exit":
            _err(
                "V542_RUN_MATRIX",
                f"statistics.holm_family.{exit_name} must be alpha=0.025, "
                "correction=holm, scope=eligible_candidate_set_within_exit "
                f"(per-exit Holm over the eligible candidate set), got {fam!r}",
            )
    proc = holm.get("process_route")
    if not isinstance(proc, Mapping) or float(proc.get("alpha") or 0) != 0.025 \
            or proc.get("correction") != "holm" or \
            proc.get("family") != "single_process_family" or \
            proc.get("scope") != "2_endpoints_x_eligible_candidates":
        _err(
            "V542_RUN_MATRIX",
            "statistics.holm_family.process_route must keep "
            "family=single_process_family (ONE Holm family alpha=0.025 over "
            f"2 endpoints x eligible candidates), got {proc!r}",
        )
    eligible = str(stats.get("h3_eligible_candidates") or "")
    if "H2" not in eligible or "diagnostic" not in eligible or \
            "三 seed" not in eligible or "winner-eligible" not in eligible:
        _err(
            "V542_RUN_MATRIX",
            "statistics.h3_eligible_candidates must require three-seed "
            "completion + per-seed H2 no-breach + winner-eligible status and "
            f"exclude diagnostic variants (A1-C/A5-C/A0), got {eligible!r}",
        )
    champ = str(stats.get("champion_rule") or "")
    if "adjusted p" not in champ:
        _err(
            "V542_RUN_MATRIX",
            "statistics.champion_rule must require adjusted p < alpha plus the "
            f"effect/non-inferiority conditions, got "
            f"{stats.get('champion_rule')!r}",
        )
    if FINAL_SELECTION_NAME not in champ:
        _err(
            "V542_RUN_MATRIX",
            f"statistics.champion_rule must route the final choice through "
            f"evaluation_contract.{FINAL_SELECTION_NAME} (qualification is "
            f"not selection), got {stats.get('champion_rule')!r}",
        )
    if "未校正" not in champ and "uncorrected" not in champ.lower():
        _err(
            "V542_RUN_MATRIX",
            "statistics.champion_rule must require reporting both uncorrected "
            "(未校正) and Holm-adjusted results, got "
            f"{stats.get('champion_rule')!r}",
        )
    agg = ec.get("seed_aggregation") if isinstance(ec, Mapping) else None
    if not isinstance(agg, Mapping):
        _err(
            "V542_MISSING_FIELD", "evaluation_contract.seed_aggregation missing")
    if agg.get("min_seeds_same_direction") != 2 or agg.get("total_seeds") != 3:
        _err(
            "V542_RUN_MATRIX",
            "seed_aggregation must carry NUMERIC min_seeds_same_direction=2 and "
            f"total_seeds=3 (string forms are rejected), got "
            f"{agg.get('min_seeds_same_direction')!r}/"
            f"{agg.get('total_seeds')!r}",
        )
    if agg.get("per_seed_h2_breach_forbidden") is not True:
        _err(
            "V542_RUN_MATRIX",
            "seed_aggregation.per_seed_h2_breach_forbidden must be true",
        )
    # the gates prose must carry the frozen statistics tokens
    gates = cfg.get("gates") or {}
    h3_rule = str(gates.get("H3", {}).get("rule") or "") \
        if isinstance(gates.get("H3"), Mapping) else ""
    for token in (str(BOOTSTRAP_SEED_ROOT), "Holm", str(FINAL_DRAWS)):
        if token not in h3_rule:
            _err(
                "V542_RUN_MATRIX",
                f"gates.H3.rule must carry the frozen statistics token {token!r} "
                "(bootstrap seed root / Holm family / final draws)",
            )


def _check_route_logic(cfg: Mapping[str, Any]) -> None:
    """fix-6 C2: structured parenthesized route logic (Oracle finding 2).

    ``move_route``/``process_route`` must carry the exact frozen
    ``pass_logic`` strings plus their component definitions, the seed17
    screen must reuse the same parenthesized logic, and the H3 gate prose
    must restate both parenthesized expressions (no dangling OR).
    """
    ec = cfg.get("evaluation_contract")
    rl = ec.get("route_logic") if isinstance(ec, Mapping) else None
    if not isinstance(rl, Mapping):
        _err(
            "V542_MISSING_FIELD",
            "evaluation_contract.route_logic missing (fix-6 C2: structured "
            "parenthesized route logic replaces ambiguous prose)",
        )
    note = str(rl.get("note") or "")
    if "括号" not in note and "parenthesized" not in note:
        _err(
            "V542_RUN_MATRIX",
            "route_logic.note must state the parenthesized-logic rule "
            f"(non-inferiority stays inside its own branch), got {note!r}",
        )
    for route_name, pass_logic in (
            ("move_route", MOVE_PASS_LOGIC),
            ("process_route", PROCESS_PASS_LOGIC)):
        route = rl.get(route_name)
        if not isinstance(route, Mapping):
            _err(
                "V542_MISSING_FIELD",
                f"route_logic.{route_name} missing",
            )
        if route.get("pass_logic") != pass_logic:
            _err(
                "V542_RUN_MATRIX",
                f"route_logic.{route_name}.pass_logic must be exactly "
                f"{pass_logic!r}, got {route.get('pass_logic')!r}",
            )
        for comp, want in ROUTE_COMPONENTS[route_name].items():
            got = route.get(comp)
            if got is None:
                _err(
                    "V542_MISSING_FIELD",
                    f"route_logic.{route_name}.{comp} missing (component "
                    "definitions are part of the freeze)",
                )
            if str(got).strip() != want:
                _err(
                    "V542_RUN_MATRIX",
                    f"route_logic.{route_name}.{comp} must be exactly "
                    f"{want!r}, got {got!r}",
                )
    if rl.get("seed17_screen_reuses_same_parenthesized_logic") is not True:
        _err(
            "V542_RUN_MATRIX",
            "route_logic.seed17_screen_reuses_same_parenthesized_logic must "
            "be true — the screen and H3 share ONE parenthesized logic",
        )
    # the H3 gate prose must restate both parenthesized expressions
    gates = cfg.get("gates") or {}
    h3_rule = str(gates.get("H3", {}).get("rule") or "") \
        if isinstance(gates.get("H3"), Mapping) else ""
    for logic in (MOVE_PASS_LOGIC, PROCESS_PASS_LOGIC):
        if logic not in h3_rule:
            _err(
                "V542_RUN_MATRIX",
                f"gates.H3.rule must restate the frozen parenthesized logic "
                f"{logic!r} (no dangling OR), got {h3_rule!r}",
            )
    # the seed17 point-effect routes reuse the same parenthesized logic
    pred = ec.get("seed17_screen_pass_predicate") if isinstance(ec, Mapping) \
        else None
    per = pred.get("point_effect_requirement") if isinstance(pred, Mapping) \
        else None
    if isinstance(per, Mapping):
        problems: List[str] = []
        for key, logic in (("move_route", MOVE_PASS_LOGIC),
                           ("process_route", PROCESS_PASS_LOGIC)):
            text = str(per.get(key) or "")
            if not text:
                problems.append(f"point_effect_requirement.{key} missing")
            elif logic not in text:
                problems.append(
                    f"point_effect_requirement.{key} must reuse the frozen "
                    f"parenthesized logic {logic!r}, got {text!r}"
                )
        if per.get("dangling_or_condition_forbidden") is not True:
            problems.append(
                "point_effect_requirement.dangling_or_condition_forbidden "
                "must be true"
            )
        if problems:
            _err("V542_RUN_MATRIX", "; ".join(problems))


def _check_final_selection(cfg: Mapping[str, Any]) -> None:
    """fix-6 C1: frozen final Backbone* selection rule (Oracle finding 1).

    H3-qualified candidates only; A0/diagnostic excluded; 0 -> A0 retained;
    1 -> that candidate; many -> the exact 7-step lexicographic tuple;
    ``dev_used=false``; report every qualified candidate; the rule block
    rides in the statistics SHA; A5-C uses the selected candidate only.
    """
    ec = cfg.get("evaluation_contract")
    rule = ec.get("final_selection_rule") if isinstance(ec, Mapping) else None
    if not isinstance(rule, Mapping):
        _err(
            "V542_MISSING_FIELD",
            "evaluation_contract.final_selection_rule missing (fix-6 C1: a "
            "frozen tie-break is required before WP6/WP7)",
        )
    if rule.get("definition_name") != FINAL_SELECTION_NAME:
        _err(
            "V542_RUN_MATRIX",
            f"final_selection_rule.definition_name must be "
            f"{FINAL_SELECTION_NAME!r}, got {rule.get('definition_name')!r}",
        )
    eligible = str(rule.get("eligible") or "")
    if "H3-qualified" not in eligible:
        _err(
            "V542_RUN_MATRIX",
            "final_selection_rule.eligible must restrict the pool to "
            f"H3-qualified candidates only, got {eligible!r}",
        )
    if tuple(rule.get("excluded") or ()) != FINAL_SELECTION_EXCLUDED:
        _err(
            "V542_RUN_MATRIX",
            "final_selection_rule.excluded must be exactly "
            f"{list(FINAL_SELECTION_EXCLUDED)} (A0 baseline + diagnostic "
            f"variants never compete), got {rule.get('excluded')!r}",
        )
    single = str(rule.get("single_candidate_rule") or "")
    if "1" not in single or "Backbone" not in single:
        _err(
            "V542_RUN_MATRIX",
            "final_selection_rule.single_candidate_rule must state that exactly "
            f"one H3-qualified candidate becomes Backbone*, got {single!r}",
        )
    none_rule = str(rule.get("no_candidate_rule") or "")
    if "0" not in none_rule or "A0" not in none_rule or \
            "no_backbone_winner" not in none_rule:
        _err(
            "V542_RUN_MATRIX",
            "final_selection_rule.no_candidate_rule must state that zero "
            "H3-qualified candidates leave A0 in place with outcome "
            f"no_backbone_winner, got {none_rule!r}",
        )
    if rule.get("dev_used") is not False:
        _err(
            "V542_RUN_MATRIX",
            "final_selection_rule.dev_used must be false — dev is a "
            "post-freeze evaluation only and never participates in selection "
            f"(got {rule.get('dev_used')!r})",
        )
    if rule.get("report_all_qualified") is not True:
        _err(
            "V542_RUN_MATRIX",
            "final_selection_rule.report_all_qualified must be true — every "
            "H3-qualified candidate is reported, not just the winner",
        )
    tie = tuple(rule.get("tie_break_order") or ())
    if tie != FINAL_SELECTION_TIE_BREAK:
        _err(
            "V542_RUN_MATRIX",
            "final_selection_rule.tie_break_order must be exactly the frozen "
            f"lexicographic tuple {list(FINAL_SELECTION_TIE_BREAK)}, got "
            f"{list(tie)!r}",
        )
    margin = rule.get("effect_margin_definition")
    if not isinstance(margin, Mapping) or dict(margin) != FINAL_EFFECT_MARGIN:
        _err(
            "V542_RUN_MATRIX",
            "final_selection_rule.effect_margin_definition must be exactly "
            f"{FINAL_EFFECT_MARGIN}, got "
            f"{dict(margin) if isinstance(margin, Mapping) else margin!r}",
        )
    if rule.get("rule_block_in_statistics_sha") is not True:
        _err(
            "V542_RUN_MATRIX",
            "final_selection_rule.rule_block_in_statistics_sha must be true — "
            "the full rule block is hashed into statistics/final-selection "
            "SHA and run identity",
        )
    if rule.get("a5_c_activation_uses_selected_candidate_only") is not True:
        _err(
            "V542_RUN_MATRIX",
            "final_selection_rule.a5_c_activation_uses_selected_candidate_only "
            "must be true — A5-C activates ONLY on a5_selected_backbone",
        )
    # the A5-C activation branch itself must stay keyed on the selection
    rm = cfg.get("run_matrix") or {}
    a5c = rm.get("a5c_activation") if isinstance(rm, Mapping) else None
    if isinstance(a5c, Mapping) and a5c.get("condition") != A5C_ACTIVATION:
        _err(
            "V542_RUN_MATRIX",
            f"run_matrix.a5c_activation.condition must be "
            f"{A5C_ACTIVATION!r} (selected candidate only), got "
            f"{a5c.get('condition')!r}",
        )
    # champion_rule must hand the final choice to this rule
    stats = ec.get("statistics") if isinstance(ec, Mapping) else None
    champ = str(stats.get("champion_rule") or "") if isinstance(stats, Mapping) \
        else ""
    if FINAL_SELECTION_NAME not in champ:
        _err(
            "V542_RUN_MATRIX",
            f"statistics.champion_rule must reference "
            f"evaluation_contract.{FINAL_SELECTION_NAME}, got {champ!r}",
        )


def _check_a0_sanity_band(cfg: Mapping[str, Any]) -> None:
    """fix-6 C5: A0 sanity band is a hard T1/top2 precondition (finding 5).

    ``V542-A0-sanity-band`` must be a registry gate referenced by
    ``t1_start_requires`` and the top2 freeze; the formal stage order puts
    T0 A0x3 before any T1 formal; a breach strictly blocks T1 (the breach
    procedure has NO exception-signing escape hatch).
    """
    a0 = cfg.get("a0_baseline")
    if not isinstance(a0, Mapping):
        _err("V542_MISSING_FIELD", "a0_baseline must be a mapping")
    if a0.get("sanity_band_gate") != A0_SANITY_BAND_GATE:
        _err(
            "V542_RUN_MATRIX",
            f"a0_baseline.sanity_band_gate must be {A0_SANITY_BAND_GATE!r}, "
            f"got {a0.get('sanity_band_gate')!r}",
        )
    band = a0.get("sanity_band_vs_v541_e0")
    if not isinstance(band, Mapping) or dict(band) != A0_SANITY_BAND_VALUES:
        _err(
            "V542_RUN_MATRIX",
            "a0_baseline.sanity_band_vs_v541_e0 must be exactly "
            f"{A0_SANITY_BAND_VALUES}, got "
            f"{dict(band) if isinstance(band, Mapping) else band!r}",
        )
    stat = str(a0.get("sanity_band_statistic") or "")
    if "A0×3" not in stat or "V541 E0×3" not in stat or \
            "seed-mean" not in stat:
        _err(
            "V542_RUN_MATRIX",
            "a0_baseline.sanity_band_statistic must compare the A0×3 "
            "three-seed MEAN against the V541 E0×3 three-seed mean "
            f"(seed-mean vs seed-mean, not one checkpoint), got {stat!r}",
        )
    order = str(a0.get("sanity_band_order") or "")
    if "T0" not in order or "T1" not in order:
        _err(
            "V542_RUN_MATRIX",
            "a0_baseline.sanity_band_order must sequence T0 A0x3 + band "
            f"before any T1 formal, got {order!r}",
        )
    steps = tuple(_normalize_steps(a0.get("sanity_band_breach_procedure") or ()))
    if steps != A0_BREACH_PROCEDURE:
        _err(
            "V542_RUN_MATRIX",
            "a0_baseline.sanity_band_breach_procedure must be exactly the "
            f"frozen {len(A0_BREACH_PROCEDURE)}-step procedure "
            f"{list(A0_BREACH_PROCEDURE)} (step 5 forbids exception "
            f"signing), got {list(steps)!r}",
        )
    # gate-0 A8: the note is frozen EXACTLY — one rule, no contradiction
    # with the breach procedure (per-defect-class identity bump; rerun
    # A0x3; exception signing still forbidden).
    note = str(a0.get("sanity_band_note") or "")
    if note != A0_SANITY_BAND_NOTE_EXACT:
        _err(
            "V542_RUN_MATRIX",
            "a0_baseline.sanity_band_note must be exactly the frozen "
            f"per-defect identity-bump rule (bans exception signing under "
            f"breach) {A0_SANITY_BAND_NOTE_EXACT!r}, got {note!r}",
        )
    # formal stage order: T0 first, T1 second, T-ATTR last
    rm = cfg.get("run_matrix")
    fso = rm.get("formal_stage_order") if isinstance(rm, Mapping) else None
    if not isinstance(fso, Mapping):
        _err(
            "V542_MISSING_FIELD",
            "run_matrix.formal_stage_order missing (fix-6 C5: frozen "
            "formal execution order)",
        )
    got_order = tuple(fso.get("order") or ())
    if got_order != FORMAL_STAGE_ORDER:
        _err(
            "V542_RUN_MATRIX",
            "run_matrix.formal_stage_order.order must be exactly "
            f"{list(FORMAL_STAGE_ORDER)} (T0 A0x3 + sanity band BEFORE any "
            f"T1 formal), got {list(got_order)!r}",
        )
    # top2 freeze requires BOTH A0_S17 and the sanity band
    top2 = rm.get("top2_preregistration") if isinstance(rm, Mapping) else None
    if isinstance(top2, Mapping):
        freeze = str(top2.get("freeze_requires") or "")
        if A0_SANITY_BAND_GATE not in freeze:
            _err(
                "V542_RUN_MATRIX",
                "top2_preregistration.freeze_requires must include the "
                f"{A0_SANITY_BAND_GATE} pass alongside A0_S17 completion, "
                f"got {freeze!r}",
            )
    pred = (cfg.get("evaluation_contract") or {}).get(
        "seed17_screen_pass_predicate")
    if isinstance(pred, Mapping):
        freeze = str(pred.get("top2_freeze_requires") or "")
        if A0_SANITY_BAND_GATE not in freeze:
            _err(
                "V542_RUN_MATRIX",
                "seed17_screen_pass_predicate.top2_freeze_requires must "
                f"include {A0_SANITY_BAND_GATE}, got {freeze!r}",
            )


def _check_seed17_screen(cfg: Mapping[str, Any]) -> None:
    """v1.1 frozen seed17_screen_pass predicate (Oracle finding 2).

    This is a seed17 SCREEN — it is NOT a three-seed H3 pass.  The predicate
    fields, hard gates, matched-A0 dependency, H2 no-breach wording and the
    A0_S17/A0_S29/S43 completion dependencies are all machine-checked.
    fix-6 C6 adds point-effect thresholds, the reused parenthesized
    route logic, the A0x3 sanity-band freeze and the fewer-than-2 rule.
    Correction (gate-1 C5): the H2 sign threshold inside
    ``h2_no_breach`` is a TOKEN check (one ``<=0.01`` token suffices —
    the dead guard supplies it, so a sign threshold of ``<=0.02`` slips
    through); it is NOT an exact threshold assertion.  Exact enforcement
    is open gate-0 advisory A2 (close before P3/WP6).
    """
    ec = cfg.get("evaluation_contract")
    pred = ec.get("seed17_screen_pass_predicate") if isinstance(ec, Mapping) else None
    if not isinstance(pred, Mapping):
        _err(
            "V542_MISSING_FIELD",
            "evaluation_contract.seed17_screen_pass_predicate missing",
        )
    if pred.get("definition_name") != "seed17_screen_pass":
        _err(
            "V542_RUN_MATRIX",
            "seed17_screen_pass_predicate.definition_name must be "
            f"'seed17_screen_pass', got {pred.get('definition_name')!r}",
        )
    scope = str(pred.get("scope") or "")
    if "H3" not in scope or ("不构成" not in scope and "NOT" not in scope):
        _err(
            "V542_RUN_MATRIX",
            "seed17_screen_pass_predicate.scope must state that this is a "
            f"seed17 screen and NOT a three-seed H3 claim, got {scope!r}",
        )
    hard = tuple(pred.get("hard_gates_all_pass") or ())
    if hard != ("H0_legality", "H1_resource", "diagnostic_gate", "cost_gate"):
        _err(
            "V542_RUN_MATRIX",
            "seed17_screen_pass_predicate.hard_gates_all_pass must be exactly "
            "['H0_legality', 'H1_resource', 'diagnostic_gate', 'cost_gate'], "
            f"got {list(hard)!r}",
        )
    matched = str(pred.get("matched_a0_dependency") or "")
    if "V542_A0_S17" not in matched or "有限" not in matched:
        _err(
            "V542_RUN_MATRIX",
            "seed17_screen_pass_predicate.matched_a0_dependency must require "
            "the matched V542_A0_S17 run to be complete with finite metrics, "
            f"got {matched!r}",
        )
    h2 = str(pred.get("h2_no_breach") or "")
    if "H2" not in h2 or "退化" not in h2:
        _err(
            "V542_RUN_MATRIX",
            "seed17_screen_pass_predicate.h2_no_breach must restate the H2 "
            f"guards with the degradation (退化) direction word, got {h2!r}",
        )
    # fix-6 C6: the four H2 guard thresholds are frozen numbers
    for tok in ("<=2%", "<=0.01", "sign", "dead", "fixed-coverage direction"):
        if tok not in h2:
            _err(
                "V542_RUN_MATRIX",
                f"seed17_screen_pass_predicate.h2_no_breach must carry the "
                f"frozen H2 guard token {tok!r} (degradation <=2%, sign "
                f"<=0.01, dead <=0.01, direction <=2%), got {h2!r}",
            )
    run_status = str(pred.get("run_status") or "")
    if "done" not in run_status or "有限" not in run_status:
        _err(
            "V542_RUN_MATRIX",
            "seed17_screen_pass_predicate.run_status must require the run "
            f"done AND every preregistered metric finite, got {run_status!r}",
        )
    per = pred.get("point_effect_requirement")
    if not isinstance(per, Mapping) or not per.get("process_route") or \
            not per.get("move_route"):
        _err(
            "V542_RUN_MATRIX",
            "seed17_screen_pass_predicate.point_effect_requirement must define "
            f"both process_route and move_route point thresholds, got {per!r}",
        )
    if isinstance(per, Mapping):
        # route_logic reuse + dangling-OR ban are enforced in
        # _check_route_logic (fix-6 C2); here we freeze the point THRESHOLDS.
        problems: List[str] = []
        rule_text = str(per.get("rule") or "")
        if "route_logic" not in rule_text:
            problems.append(
                "rule must state that the screen reuses "
                "evaluation_contract.route_logic (fix-6 C2)"
            )
        # exact point thresholds: PROCESS 0.02/0.03/0.01/0.02,
        # MOVE 3%/0.01/3%/2%/2% — the numbers are part of the freeze.
        for key, tokens in (
                ("process_route", ("0.02", "0.03", "0.01", "0.02")),
                ("move_route", ("3%", "0.01", "3%", "2%", "2%"))):
            text = str(per.get(key) or "")
            if not text:
                problems.append(f"{key} missing")
                continue
            for tok in tokens:
                if tok not in text:
                    problems.append(
                        f"{key} must carry the frozen threshold token {tok!r}"
                    )
        if problems:
            _err(
                "V542_RUN_MATRIX",
                "seed17_screen_pass_predicate.point_effect_requirement: "
                + "; ".join(problems)
                + f", got {dict(per)!r}",
            )
    if "A0_S17" not in str(pred.get("top2_freeze_requires") or ""):
        _err(
            "V542_RUN_MATRIX",
            "seed17_screen_pass_predicate.top2_freeze_requires must pin "
            f"A0_S17 completion before the top-2 freeze, got "
            f"{pred.get('top2_freeze_requires')!r}",
        )
    rank = str(pred.get("ranking") or "")
    if "最多 2" not in rank or "可少于 2" not in rank or "不替补" not in rank:
        _err(
            "V542_RUN_MATRIX",
            "seed17_screen_pass_predicate.ranking must allow FEWER than two "
            "candidates with no substitution — frozen phrase "
            "'最多 2 个; 可少于 2; 不替补' (max 2, may be fewer, no "
            f"replacement), got {rank!r}",
        )
    if "screen" not in str(pred.get("definition_name") or ""):
        _err(
            "V542_RUN_MATRIX",
            "seed17_screen_pass_predicate.definition_name must remain a "
            f"screen, got {pred.get('definition_name')!r}",
        )
    h3_req = str(pred.get("h3_final_requires") or "")
    if "A0" not in h3_req or "S29" not in h3_req or "S43" not in h3_req:
        _err(
            "V542_RUN_MATRIX",
            "seed17_screen_pass_predicate.h3_final_requires must require the "
            f"matched A0 S29/S43 runs complete before final H3, got {h3_req!r}",
        )


def _check_identity_start(cfg: Mapping[str, Any]) -> None:
    """v1.1 identity/start contract (Oracle findings 10–12).

    ``source_manifest_sha256`` required / ``code_commit`` optional; exactly
    the 14 frozen identity fields; ``t1_start_requires`` resolves to registry
    canonical gate IDs plus the ``runplan_pin_record`` record whitelist.
    """
    ic = cfg.get("identity_contract")
    if not isinstance(ic, Mapping):
        _err("V542_MISSING_FIELD", "identity_contract must be a mapping")
    req = tuple(ic.get("run_identity_required") or ())
    if set(req) != set(IDENTITY_REQUIRED_FIELDS) or \
            len(req) != len(IDENTITY_REQUIRED_FIELDS):
        _err(
            "V542_MISSING_FIELD",
            "identity_contract.run_identity_required must be exactly the 14 "
            f"frozen fields {list(IDENTITY_REQUIRED_FIELDS)} "
            "(source_manifest_sha256 required; code_commit never required); "
            f"got {list(req)!r}",
        )
    opt = tuple(ic.get("run_identity_optional") or ())
    if opt != IDENTITY_OPTIONAL_FIELDS:
        _err(
            "V542_MISSING_FIELD",
            "identity_contract.run_identity_optional must be exactly "
            f"{list(IDENTITY_OPTIONAL_FIELDS)} (code_commit is record-only "
            f"under git_commit_authoritative=false), got {list(opt)!r}",
        )
    code = ic.get("code")
    if not isinstance(code, Mapping) or \
            code.get("git_commit_authoritative") is not False:
        _err(
            "V542_MISSING_FIELD",
            "identity_contract.code.git_commit_authoritative must be false — "
            "code_commit is an optional record, source_manifest_sha256 is the "
            "code identity",
        )
    gov = cfg.get("governance")
    auth = gov.get("authorization") if isinstance(gov, Mapping) else None
    requires = tuple(auth.get("t1_start_requires") or ()) \
        if isinstance(auth, Mapping) else ()
    if not requires:
        _err(
            "V542_MISSING_FIELD",
            "governance.authorization.t1_start_requires missing/empty",
        )
    if "runplan_pin_record" not in requires:
        _err(
            "V542_RUN_MATRIX",
            "t1_start_requires must include the runplan_pin_record record "
            f"requirement, got {list(requires)!r}",
        )
    try:
        gv_map = load_gv_map(DEFAULT_MAP_PATH)
        campaign = resolve_campaign(gv_map, GV_CAMPAIGN_ID)
    except GVMapError as exc:
        _err("V542_GV_MISMATCH", str(exc))
    gate_ids = set(campaign.gates)
    for entry in requires:
        if entry in T1_RECORD_WHITELIST:
            continue
        if entry not in gate_ids:
            _err(
                "V542_RUN_MATRIX",
                f"t1_start_requires entry {entry!r} is neither a registry "
                f"canonical gate ID (g_v_map.yaml V5.4.2 gates) nor a "
                f" whitelisted record {sorted(T1_RECORD_WHITELIST)}; legacy "
                "rule keys (H0/H1) and aliases (runplan_pin) are forbidden",
            )
    # fix-6: the canonical set is frozen EXACTLY (order + membership) and
    # every canonical gate must exist in the G-V registry (incl. the A0
    # sanity band, which is a hard T1 precondition).
    if requires != T1_START_REQUIRES:
        _err(
            "V542_RUN_MATRIX",
            "governance.authorization.t1_start_requires must be exactly "
            f"{list(T1_START_REQUIRES)} (fix-6: A0 sanity band is a hard T1 "
            f"precondition), got {list(requires)}",
        )
    for canonical in T1_CANONICAL_GATES:
        if canonical not in gate_ids:
            _err(
                "V542_GV_MISMATCH",
                f"canonical T1 gate {canonical!r} is not registered in "
                "g_v_map.yaml V5.4.2 gates",
            )
        if canonical not in requires:
            _err(
                "V542_RUN_MATRIX",
                f"t1_start_requires must include the canonical gate "
                f"{canonical!r}, got {list(requires)!r}",
            )
    if A0_SANITY_BAND_GATE not in requires:
        _err(
            "V542_RUN_MATRIX",
            f"t1_start_requires must include the A0 sanity band gate "
            f"{A0_SANITY_BAND_GATE!r} (fix-6 C5: hard T1 precondition), "
            f"got {list(requires)!r}",
        )
    # profile conditional variants must cover both attribution diagnostics
    # (checked in _check_dose_blocks); budget entries are checked in
    # validate_config once the table is filled.


def _check_thresholds(cfg: Mapping[str, Any], *, formal: bool) -> List[str]:
    th = cfg.get("thresholds")
    if not isinstance(th, Mapping):
        _err("V542_MISSING_FIELD", "thresholds must be a mapping")
    # fix-6: the signed threshold set is frozen at exactly 28 keys (the
    # original 24 + bootstrap_seed_root/top2_max_runs/a5_supplement_max_runs/
    # t3_max_runs) and the config block may carry no extra/renamed keys.
    if len(FORMAL_THRESHOLDS) != 28:  # pragma: no cover - frozen constant
        _err(
            "V542_THRESHOLD_NULL",
            f"internal invariant broken: FORMAL_THRESHOLDS must stay at 28 "
            f"signed keys, got {len(FORMAL_THRESHOLDS)}",
        )
    signed_keys = {k for k in th if k != "signoff"}
    if signed_keys != set(FORMAL_THRESHOLDS):
        missing = sorted(set(FORMAL_THRESHOLDS) - signed_keys)
        extra = sorted(signed_keys - set(FORMAL_THRESHOLDS))
        _err(
            "V542_THRESHOLD_NULL",
            "thresholds block must contain exactly the 28 signed keys "
            f"(missing={missing} extra={extra})",
        )
    nulls = [k for k in FORMAL_THRESHOLDS if th.get(k) is None]
    if formal:
        if nulls:
            _err(
                "V542_THRESHOLD_NULL",
                f"formal gate: threshold(s) still null {nulls}; unsigned values "
                "block the launch (runner must not fill defaults)",
            )
        signoff = th.get("signoff")
        if not isinstance(signoff, Mapping) or signoff.get("status") != "frozen":
            _err(
                "V542_THRESHOLD_NULL",
                "formal gate: thresholds.signoff.status must be 'frozen'",
            )
        for key in FORMAL_THRESHOLDS:
            if signoff.get(key) != th.get(key):
                _err(
                    "V542_THRESHOLD_NULL",
                    f"formal gate: thresholds.signoff.{key} "
                    f"({signoff.get(key)!r}) != thresholds.{key} ({th.get(key)!r}); "
                    "signed record must mirror the config values byte-for-byte",
                )
    for key in FORMAL_THRESHOLDS:
        if key in th and th[key] is not None and not isinstance(
                th[key], (int, float)):
            _err(
                "V542_THRESHOLD_NULL",
                f"threshold {key!r} must be numeric, got {th[key]!r}",
            )
    # parameter invariants must agree with the signed thresholds
    if th.get("a0_backbone_active") is not None and \
            th.get("a0_backbone_active") != A0_ACTIVE_PARAMS:
        _err(
            "V542_PARAM_BUDGET",
            "thresholds.a0_backbone_active must be 55392",
        )
    if th.get("a0_joint_params") is not None and \
            th.get("a0_joint_params") != A0_JOINT_PARAMS:
        _err(
            "V542_PARAM_BUDGET", "thresholds.a0_joint_params must be 57619")
    if th.get("move_decoder_params") is not None and \
            th.get("move_decoder_params") != DECODER_PARAMS:
        _err("V542_PARAM_BUDGET", "thresholds.move_decoder_params must be 49")
    if th.get("process_head_params") is not None and \
            th.get("process_head_params") != PROCESS_PARAMS:
        _err("V542_PARAM_BUDGET", "thresholds.process_head_params must be 2178")
    if th.get("backbone_active_max") is not None and \
            th.get("backbone_active_max") != BACKBONE_ACTIVE_MAX:
        _err(
            "V542_PARAM_BUDGET",
            f"thresholds.backbone_active_max must be {BACKBONE_ACTIVE_MAX}",
        )
    # v1.1 statistics/budget thresholds mirror the frozen constants
    if th.get("bootstrap_seed_root") is not None and \
            th.get("bootstrap_seed_root") != BOOTSTRAP_SEED_ROOT:
        _err(
            "V542_THRESHOLD_NULL",
            f"thresholds.bootstrap_seed_root must be {BOOTSTRAP_SEED_ROOT}",
        )
    if th.get("final_bootstrap_draws") is not None and \
            th.get("final_bootstrap_draws") != FINAL_DRAWS:
        _err(
            "V542_THRESHOLD_NULL",
            f"thresholds.final_bootstrap_draws must be {FINAL_DRAWS}",
        )
    if th.get("top2_max_runs") is not None and \
            th.get("top2_max_runs") != TOP2_MAX_RUNS:
        _err(
            "V542_RUN_MATRIX",
            f"thresholds.top2_max_runs must be {TOP2_MAX_RUNS}",
        )
    if th.get("a5_supplement_max_runs") is not None and \
            th.get("a5_supplement_max_runs") != A5_SUPPLEMENT_MAX_RUNS:
        _err(
            "V542_RUN_MATRIX",
            f"thresholds.a5_supplement_max_runs must be {A5_SUPPLEMENT_MAX_RUNS}",
        )
    if th.get("t3_max_runs") is not None and \
            th.get("t3_max_runs") != T3_MAX_RUNS:
        _err(
            "V542_RUN_MATRIX",
            f"thresholds.t3_max_runs must be {T3_MAX_RUNS}",
        )
    # H2 must keep the degradation (退化) direction word (Oracle finding 7)
    gates = cfg.get("gates") or {}
    h2 = gates.get("H2")
    h2_rule = str(h2.get("rule") or "") if isinstance(h2, Mapping) else ""
    if "退化" not in h2_rule:
        _err(
            "V542_RUN_MATRIX",
            "gates.H2.rule must carry the degradation (退化) qualifier — H2 is "
            f"a non-inferiority guard (degradation <=2%), got {h2_rule!r}",
        )
    return nulls


def _check_run_outcomes(cfg: Mapping[str, Any]) -> None:
    ro = cfg.get("run_outcomes")
    if not isinstance(ro, Mapping):
        _err("V542_MISSING_FIELD", "run_outcomes must be a mapping")
    vocab = ro.get("vocabulary")
    if not isinstance(vocab, Sequence) or set(vocab) != OUTCOME_VOCABULARY:
        _err(
            "V542_RUN_MATRIX",
            f"run_outcomes.vocabulary must be exactly {sorted(OUTCOME_VOCABULARY)}, "
            f"got {vocab!r}",
        )
    if ro.get("ledger_status_machine") != "unchanged":
        _err(
            "V542_RUN_MATRIX",
            "run_outcomes.ledger_status_machine must be 'unchanged' — V542 run "
            "outcome is tracked separately from the legacy ledger status machine",
        )


def _check_cost_acceptance(cfg: Mapping[str, Any]) -> None:
    ca = cfg.get("cost_acceptance")
    if not isinstance(ca, Mapping):
        _err("V542_MISSING_FIELD", "cost_acceptance must be a mapping")
    if ca.get("rule") != "hard_ratio":
        _err(
            "V542_RUN_MATRIX", "cost_acceptance.rule must be 'hard_ratio'")
    denom = str(ca.get("denominator") or "")
    if "V542" not in denom or "A0" not in denom:
        _err(
            "V542_RUN_MATRIX",
            "cost_acceptance.denominator must be the V542 A0 same-env profile "
            f"measurement, got {denom!r}",
        )
    ratio = str(ca.get("ratio_expression") or "")
    if "max_core_time_ratio" not in ratio:
        _err(
            "V542_RUN_MATRIX",
            "cost_acceptance.ratio_expression must reference "
            "thresholds.max_core_time_ratio",
        )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def validate_config(
    cfg: Mapping[str, Any],
    *,
    formal: bool = False,
    check_gv: bool = True,
) -> Dict[str, Any]:
    """Run every check; return a machine-readable report.

    Raises :class:`V542SchemaError` on the first typed rejection.
    """
    checks: List[str] = []

    _check_schema(cfg)
    checks.append("schema")
    _check_objective_and_arms(cfg)
    checks.append("objective")
    _check_required_fields(cfg)
    checks.append("required_fields")
    _check_backbone_registry(cfg)
    checks.append("backbone_registry")
    _check_forward_view(cfg)
    checks.append("forward_view")
    _check_interface(cfg)
    checks.append("interface")
    _check_param_budget(cfg)
    checks.append("param_budget")
    init_blocked = _check_init_source(cfg, formal=formal)
    checks.append("init_source")
    _check_a0_decision(cfg)
    checks.append("a0_decision")
    _check_authorization(cfg)
    checks.append("authorization")
    _check_sampler_binding(cfg)
    checks.append("sampler_binding")
    _check_dose_blocks(cfg)
    checks.append("dose_blocks")
    _check_selection_tau(cfg)
    checks.append("selection_tau")
    if check_gv:
        _check_gv_block(cfg)
        checks.append("gv_registry")
    _check_identity_start(cfg)
    checks.append("identity_start")
    _check_run_matrix(cfg)
    checks.append("run_matrix")
    runs = cfg.get("run_matrix").get("runs")
    _check_top2_and_a5_supplement(cfg, runs)
    checks.append("top2_preregistration")
    _check_activation_branches(cfg, runs)
    checks.append("activation_branches")
    _check_statistics(cfg)
    checks.append("statistics")
    _check_route_logic(cfg)
    checks.append("route_logic")
    _check_final_selection(cfg)
    checks.append("final_selection")
    _check_seed17_screen(cfg)
    checks.append("seed17_screen")
    _check_a0_sanity_band(cfg)
    checks.append("a0_sanity_band")
    _check_run_outcomes(cfg)
    checks.append("run_outcomes")
    _check_cost_acceptance(cfg)
    checks.append("cost_acceptance")
    null_thresholds = _check_thresholds(cfg, formal=formal)
    checks.append("thresholds")

    t1_blocked: List[str] = []
    t1_blocked.extend(init_blocked)
    if null_thresholds:
        t1_blocked.append(
            f"unsigned threshold(s) {null_thresholds} block the launch"
        )
    budget = cfg.get("budget")
    if isinstance(budget, Mapping):
        table = budget.get("per_candidate_core_time")
        if table is None:
            if formal:
                _err(
                    "V542_BUDGET_NULL",
                    "formal gate: budget.per_candidate_core_time is still null — "
                    "the same-env A0 profile must be measured and the budget signed "
                    "before any formal run (prereg mode reports this as t1_blocked; "
                    "the runner must never fill defaults)",
                )
            t1_blocked.append(
                "budget.per_candidate_core_time null: same-env 40+40 W0/W1 profile "
                "signoff required before formal launch"
            )
        elif isinstance(table, Mapping):
            # v1.1: the attribution diagnostics are full-dose runs and need
            # their own budget entries before they can activate.
            missing_budget = sorted(KNOWN_BACKBONES - set(table))
            if missing_budget:
                _err(
                    "V542_BUDGET_NULL",
                    "budget.per_candidate_core_time must cover every backbone "
                    f"(incl. the A1-C/A5-C full-dose diagnostics); missing "
                    f"{missing_budget}",
                )

    return {
        "ok": True,
        "schema": SCHEMA_ID,
        "campaign_id": cfg.get("campaign_id"),
        "objective": cfg.get("objective"),
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
        "budget table (V542_BUDGET_NULL typed rejection while null)",
    )
    parser.add_argument("--json", action="store_true", help="print report as JSON")
    parser.add_argument(
        "--no-gv", action="store_true", help="skip the G–V registry cross-check"
    )
    args = parser.parse_args(argv)
    try:
        report = validate_config(
            load_config(args.config), formal=args.formal, check_gv=not args.no_gv
        )
    except V542SchemaError as exc:
        if args.json:
            print(json.dumps({"ok": False, "code": exc.code, "detail": exc.detail}))
        else:
            print(f"V542 config rejected: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            "V542 config OK: schema={schema} objective={objective} mode={mode} "
            "checks={n}".format(n=len(report["checks"]), **report)
        )
        if report["t1_blocked"]:
            print("Formal launch not config-ready:")
            for reason in report["t1_blocked"]:
                print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
