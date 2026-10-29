#!/usr/bin/env python3
"""V5.4.1 campaign config schema validator (WP0, ADR-0032).

Validates ``config/campaigns/pes2ts_v541_embed_50k.yaml`` (schema
``pes2ts_v541_campaign_v1``) and refuses to let a runner start from an
invalid config.  Deliberately independent from the V5.3-MP validator inside
``scripts/campaigns/r3_v53mp_run.py`` — V5.3 files are untouched; this module
only reads the V541 yaml plus the frozen G–V registry and the forward-view
whitelist contract.

Typed rejections (each carries a stable ``code``):

============================  ==========================================================
code                          rejection
============================  ==========================================================
``V541_CONFIG_MISSING``       file absent / unreadable / not a mapping
``V541_SCHEMA``               top-level ``schema`` != ``pes2ts_v541_campaign_v1``
``V541_MISSING_FIELD``        a required field is absent or None
``V541_OBJECTIVE``            ``objective`` != ``unified_move_process`` (only legal name)
``V541_ARM_MATRIX``           legacy three-arm matrix exposed (``arm``/``arms`` keys)
``V541_EMBEDDING_UNKNOWN``    embedding id outside the frozen E0–E6 registry
``V541_FORWARD_KEY``          forward-view key outside the contract whitelist
``V541_EDGE_ADAPTER``         ``edge_adapter`` requested (forbidden this stage)
``V541_NODE_VECTOR``          core declares vector consumption (forbidden this stage)
``V541_CONFIRM``              confirm/sealed/QC authorization not false/0
``V541_RESOURCE``             single-lane / swap / OOM guards violated
``V541_THRESHOLD_NULL``       a §11.2 threshold is null while ``--formal`` requested
``V541_GV_MISMATCH``          ``gv`` block disagrees with ``config/registry/g_v_map.yaml``
``V541_RUN_MATRIX``           run id / class / seed outside the preregistered shape
``V541_SAMPLER_DRIFT``        ``sampler_binding`` != the live ``v541_strata`` spec (id/SHA/strata/rules)
``V541_FEATURE_CONST_DRIFT``  chem-edge dim / angle tie / RBF constants != live provider code
``V541_IDENTITY_CONTRACT``    code-fingerprint / common-protocol / environment contract broken
``V541_R3_INIT``              R3 init path/SHA absent, malformed, drifted or unverifiable
``V541_DEGREE_STATS_FIT``     E2 degree stats not pinned to the inner-train-only fit
``V541_CODE_CONSTANTS``       live spec constants cannot be imported (cross-check impossible)
============================  ==========================================================

Modes
-----
* ``prereg`` (default) — structural validation; null thresholds are reported
  as ``t1_blocked`` reasons instead of hard errors (plan §3.3: unsigned stays
  ``null`` and blocks T1, the runner never fills defaults).
* ``formal`` — additionally requires every §11.2 threshold to be a non-null
  number and the signoff block to be frozen, i.e. this is the gate a formal
  T1 launch must pass.

CLI::

    python scripts/agent_ops/v541_config_schema.py                 # prereg mode
    python scripts/agent_ops/v541_config_schema.py --formal        # formal gate
    python scripts/agent_ops/v541_config_schema.py --config PATH --json
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
    "CAMPAIGN_ID",
    "DEFAULT_CONFIG",
    "EMBEDDING_REGISTRY",
    "GV_CAMPAIGN_ID",
    "KNOWN_EMBEDDINGS",
    "OBJECTIVE",
    "REQUIRED_FIELDS",
    "SCHEMA_ID",
    "V541SchemaError",
    "load_config",
    "main",
    "validate_config",
]

SCHEMA_ID = "pes2ts_v541_campaign_v1"
CAMPAIGN_ID = "PES2TS_V541_EMBED_50K"
GV_CAMPAIGN_ID = "V5.4.1"
OBJECTIVE = "unified_move_process"

#: Frozen embedding registry (plan §6.1): id -> (in_main_matrix, family).
EMBEDDING_REGISTRY: Dict[str, Tuple[bool, str]] = {
    "E0": (True, "legacy"),
    "E1": (True, "gin_e_chemical"),
    "E2": (True, "pna_chemical"),
    "E3": (True, "schnet_spatial"),
    "E4": (True, "angle_mpnn"),
    "E5": (False, "dual_stream"),
    "E6": (False, "painn_scalar"),
}
KNOWN_EMBEDDINGS = frozenset(EMBEDDING_REGISTRY)

#: §11.2 thresholds that must be non-null before formal T1 (null blocks T1).
FORMAL_THRESHOLDS: Tuple[str, ...] = (
    "sign_drop_abs_max",
    "dead_fraction_increase_abs_max",
    "fixed_coverage_direction_regression_rel_max_pct",
    "other_endpoint_auprc_drop_abs_max",
    "other_q_corr_drop_abs_max",
    "gpu_parity_atol",
    "gpu_parity_rtol",
    "max_core_time_ratio",
    "final_bootstrap_draws",
    "rollout_large_subset_curves",
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
    "embedding_contract",
    "training_contract",
    "smoke",
    "run_matrix",
    "evaluation_contract",
    "gates",
    "thresholds",
    "cost_acceptance",
    "budget",
    "run_outcomes",
    "plan_outputs",
)

#: Top-level keys that would re-expose the legacy three-arm matrix (§3.1.4).
ARM_MATRIX_KEYS = ("arm", "arms", "arm_matrix")

#: Keys whose truthy value requests forbidden consumption (§5.2/§16.2).
EDGE_ADAPTER_KEY = "edge_adapter"
NODE_VECTOR_KEY = "node_vector_consumed"

RUN_CLASSES = frozenset({"nominal", "conditional", "diagnostic"})

#: §9.1 T3 preregistration: every candidate that CAN be selected by the
#: inner-val top-2 screen is preregistered as a conditional run (E1–E5 ×
#: seeds 29/43 = 10 stable run_ids); only the two frozen candidates actually
#: activate (max 4 runs), the rest record ``gate_skipped`` and contribute 0
#: new trainings — ``N_core`` is counted by the §9.1 formula, never by
#: listed runs.
T3_CANDIDATES: Tuple[str, ...] = ("E1", "E2", "E3", "E4", "E5")
T3_SEEDS: Tuple[int, ...] = (29, 43)
T3_FULL_ACTIVATION = 4
T3_ACTIVATION = "activated_only_by_inner_val_top2"
T3_PARENT_GATE = "parents_E1_E4_H0_H1_H2_pass_and_budget_approved"
T3_UNACTIVATED_STATUS = "gate_skipped"
T3_RUN_ID_TOKEN = "TOP2"  # ambiguous placeholder ids are forbidden

_DEFAULT_CONFIG = (
    PROJECT_ROOT / "config" / "campaigns" / "pes2ts_v541_embed_50k.yaml"
)
DEFAULT_CONFIG = _DEFAULT_CONFIG

_GV_DEFAULT = (
    "campaign_id", "primary_g", "supporting_g", "gates", "map_version"
)


class V541SchemaError(ValueError):
    """Typed rejection with a stable machine code."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Read + minimally check the V541 campaign yaml (fail-closed)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not cfg_path.is_file():
        raise V541SchemaError("V541_CONFIG_MISSING", str(cfg_path))
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V541SchemaError("V541_CONFIG_MISSING", f"{cfg_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise V541SchemaError("V541_CONFIG_MISSING", f"{cfg_path}: not a mapping")
    return raw


# ---------------------------------------------------------------------------
# individual checks — each returns None or raises V541SchemaError
# ---------------------------------------------------------------------------
def _check_schema(cfg: Mapping[str, Any]) -> None:
    if cfg.get("schema") != SCHEMA_ID:
        raise V541SchemaError(
            "V541_SCHEMA", f"expected {SCHEMA_ID!r}, got {cfg.get('schema')!r}"
        )
    if cfg.get("campaign_id") != CAMPAIGN_ID:
        raise V541SchemaError(
            "V541_SCHEMA",
            f"campaign_id expected {CAMPAIGN_ID!r}, got {cfg.get('campaign_id')!r}",
        )


def _check_required_fields(cfg: Mapping[str, Any]) -> None:
    missing = [k for k in REQUIRED_FIELDS if cfg.get(k) is None]
    if missing:
        raise V541SchemaError(
            "V541_MISSING_FIELD", f"missing/None required field(s): {missing}"
        )


def _check_objective_and_arms(cfg: Mapping[str, Any]) -> None:
    objective = cfg.get("objective")
    if objective != OBJECTIVE:
        raise V541SchemaError(
            "V541_OBJECTIVE",
            f"objective must be {OBJECTIVE!r} (only legal external task name), "
            f"got {objective!r}",
        )
    for key in ARM_MATRIX_KEYS:
        if key in cfg:
            raise V541SchemaError(
                "V541_ARM_MATRIX",
                f"top-level {key!r} re-exposes the legacy three-arm matrix; "
                "V5.4.1 config/logs/reports/run_id must stay single-objective",
            )


def _check_embedding_registry(cfg: Mapping[str, Any]) -> None:
    contract = cfg.get("embedding_contract")
    if not isinstance(contract, Mapping):
        raise V541SchemaError("V541_MISSING_FIELD", "embedding_contract must be a mapping")
    registry = contract.get("embedding_registry")
    if not isinstance(registry, Mapping) or not registry:
        raise V541SchemaError(
            "V541_MISSING_FIELD", "embedding_contract.embedding_registry missing"
        )
    unknown = sorted(set(registry) - KNOWN_EMBEDDINGS)
    if unknown:
        raise V541SchemaError(
            "V541_EMBEDDING_UNKNOWN",
            f"embedding id(s) {unknown} outside frozen E0–E6 registry",
        )
    for eid, spec in registry.items():
        if not isinstance(spec, Mapping):
            raise V541SchemaError(
                "V541_MISSING_FIELD", f"embedding_registry.{eid} must be a mapping"
            )
        for field in ("family", "internal_view", "output", "in_main_matrix"):
            if spec.get(field) is None:
                raise V541SchemaError(
                    "V541_MISSING_FIELD", f"embedding_registry.{eid}.{field} missing"
                )
        expected_matrix, expected_family = EMBEDDING_REGISTRY[eid]
        if bool(spec.get("in_main_matrix")) != expected_matrix:
            raise V541SchemaError(
                "V541_EMBEDDING_UNKNOWN",
                f"embedding_registry.{eid}.in_main_matrix must be "
                f"{expected_matrix} (registry freeze, plan §6.1)",
            )
        if spec.get("family") != expected_family:
            raise V541SchemaError(
                "V541_EMBEDDING_UNKNOWN",
                f"embedding_registry.{eid}.family must be {expected_family!r}, "
                f"got {spec.get('family')!r}",
            )


def _check_forward_view(cfg: Mapping[str, Any]) -> None:
    contract = cfg.get("embedding_contract")
    keys = contract.get("forward_view_keys") if isinstance(contract, Mapping) else None
    if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)) or not keys:
        raise V541SchemaError(
            "V541_MISSING_FIELD", "embedding_contract.forward_view_keys missing/empty"
        )
    whitelist = set(FORWARD_VIEW_KEYS_V3)
    illegal = sorted({str(k) for k in keys} - whitelist)
    if illegal:
        raise V541SchemaError(
            "V541_FORWARD_KEY",
            f"forward key(s) {illegal} not in contract whitelist "
            f"FORWARD_VIEW_KEYS_V3",
        )


def _check_core_consumption(cfg: Mapping[str, Any]) -> None:
    contract = cfg.get("embedding_contract")
    core = contract.get("core_consumption") if isinstance(contract, Mapping) else None
    if not isinstance(core, Mapping):
        raise V541SchemaError(
            "V541_MISSING_FIELD", "embedding_contract.core_consumption missing"
        )
    if core.get(EDGE_ADAPTER_KEY) is not False:
        raise V541SchemaError(
            "V541_EDGE_ADAPTER",
            f"core_consumption.{EDGE_ADAPTER_KEY} must be false this stage; "
            "EdgeAdapter-style edge consumption is not part of V5.4.1 (§5.2/§16.2)",
        )
    if core.get(NODE_VECTOR_KEY) is not False:
        raise V541SchemaError(
            "V541_NODE_VECTOR",
            f"core_consumption.{NODE_VECTOR_KEY} must be false this stage; "
            "node_vector is not consumed by the fixed core (§5.2/§16.2)",
        )
    if core.get("edge_scalar_channels") != 22:
        raise V541SchemaError(
            "V541_EDGE_ADAPTER",
            "core_consumption.edge_scalar_channels must be the legacy 22 channels "
            f"in legacy order, got {core.get('edge_scalar_channels')!r}",
        )


def _check_authorization(cfg: Mapping[str, Any]) -> None:
    gov = cfg.get("governance")
    auth = gov.get("authorization") if isinstance(gov, Mapping) else None
    if not isinstance(auth, Mapping):
        raise V541SchemaError("V541_MISSING_FIELD", "governance.authorization missing")
    for flag in ("confirm_authorized", "sealed_authorized"):
        if auth.get(flag) is not False:
            raise V541SchemaError(
                "V541_CONFIRM",
                f"authorization.{flag} must be false (confirm/sealed stay sealed), "
                f"got {auth.get(flag)!r}",
            )
    qc = auth.get("new_qc_calls")
    if qc != 0:
        raise V541SchemaError(
            "V541_CONFIRM",
            f"authorization.new_qc_calls must be 0 for V5.4.1 (no QC lane), got {qc!r}",
        )
    if auth.get("single_lane") is not True:
        raise V541SchemaError(
            "V541_RESOURCE",
            f"authorization.single_lane must be true, got {auth.get('single_lane')!r}",
        )
    if auth.get("swap_allowed") is not False:
        raise V541SchemaError(
            "V541_RESOURCE",
            f"authorization.swap_allowed must be false, got {auth.get('swap_allowed')!r}",
        )
    budget = cfg.get("budget")
    if isinstance(budget, Mapping):
        if budget.get("single_lane") is not True:
            raise V541SchemaError("V541_RESOURCE", "budget.single_lane must be true")
        if budget.get("swap_max") != 0:
            raise V541SchemaError(
                "V541_RESOURCE",
                f"budget.swap_max must be 0, got {budget.get('swap_max')!r}",
            )


# ---------------------------------------------------------------------------
# live-code cross-checks (anti-drift: the frozen config must mirror what the
# code implements TODAY — sampler spec SHA, provider constants, registry rows)
# ---------------------------------------------------------------------------

_SHA64_RE = re.compile(r"^[0-9a-f]{64}$")

#: Freshly-imported live V541 spec constants (cached after the first call).
_CODE_CONSTANTS: Dict[str, Any] = {}


def _code_constants() -> Dict[str, Any]:
    """Import the LIVE V541 spec constants (fail-closed, cached once).

    The validator compares the frozen config against the code that will
    actually run: ``v541_strata`` (sampler id / spec SHA / four strata),
    ``feature_providers`` (chem-edge dim, angle tie policy) and the WP2
    embedding registry.  If the import fails the check REFUSES — a
    validator that cannot cross-check the code never reports ``ok``.
    """
    if _CODE_CONSTANTS:
        return _CODE_CONSTANTS
    try:
        from pes2ts_core.pesn.models.v541 import embedding_registry as reg
        from pes2ts_core.pesn.models.v541 import feature_providers as prov
        from pes2ts_core.pesn.offline.training import v541_identity as ident
        from pes2ts_core.pesn.offline.training import v541_strata as strata
    except Exception as exc:  # noqa: BLE001 - fail-closed on ANY import error
        raise V541SchemaError(
            "V541_CODE_CONSTANTS",
            "cannot import the live V541 spec constants ({}); refusing to "
            "validate a config against code it cannot see".format(exc),
        ) from exc
    _CODE_CONSTANTS.update(
        strata=strata, prov=prov, reg=reg, ident=ident)
    return _CODE_CONSTANTS


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_sampler_binding(cfg: Mapping[str, Any]) -> None:
    """``training_contract.sampler_binding`` must equal the LIVE spec (§7.2)."""
    tc = cfg.get("training_contract")
    binding = tc.get("sampler_binding") if isinstance(tc, Mapping) else None
    if not isinstance(binding, Mapping):
        raise V541SchemaError(
            "V541_MISSING_FIELD", "training_contract.sampler_binding missing")
    batch = tc.get("batch") if isinstance(tc, Mapping) else None
    if not isinstance(batch, Mapping):
        raise V541SchemaError(
            "V541_MISSING_FIELD", "training_contract.batch missing")
    strata = _code_constants()["strata"]
    spec = strata.SAMPLER_SPEC_V541

    def _drift(what: str, got: Any, want: Any) -> None:
        raise V541SchemaError(
            "V541_SAMPLER_DRIFT",
            "sampler_binding.{} = {!r} != live code {!r} (plan §7.2)".format(
                what, got, want),
        )

    if binding.get("sampler_id") != strata.SAMPLER_ID_V541:
        _drift("sampler_id", binding.get("sampler_id"), strata.SAMPLER_ID_V541)
    if binding.get("spec_sha256") != strata.sampler_spec_sha256():
        _drift("spec_sha256", binding.get("spec_sha256"),
               strata.sampler_spec_sha256())
    if tuple(binding.get("strata") or ()) != strata.STRATA_ORDER_V541:
        _drift("strata", binding.get("strata"), list(strata.STRATA_ORDER_V541))
    if binding.get("batch_curves") != strata.BATCH_CURVES_V541:
        _drift("batch_curves", binding.get("batch_curves"),
               strata.BATCH_CURVES_V541)
    if binding.get("frames_per_curve") != strata.FRAMES_PER_CURVE_V541:
        _drift("frames_per_curve", binding.get("frames_per_curve"),
               strata.FRAMES_PER_CURVE_V541)
    want_dirs = dict(spec["batch_shape"]["direction_slots_per_curve"])
    if dict(binding.get("direction_balance") or {}) != want_dirs:
        _drift("direction_balance", binding.get("direction_balance"), want_dirs)
    if dict(binding.get("mirror") or {}) != dict(strata.MIRROR_STRATA_V541):
        _drift("mirror", binding.get("mirror"), dict(strata.MIRROR_STRATA_V541))
    # the batch block and the binding must not disagree with each other
    if tuple(batch.get("strata") or ()) != strata.STRATA_ORDER_V541:
        _drift("batch.strata", batch.get("strata"), list(strata.STRATA_ORDER_V541))
    if batch.get("curves") != strata.BATCH_CURVES_V541:
        _drift("batch.curves", batch.get("curves"), strata.BATCH_CURVES_V541)
    if batch.get("frames") != strata.FRAMES_PER_CURVE_V541:
        _drift("batch.frames", batch.get("frames"), strata.FRAMES_PER_CURVE_V541)
    # frozen rule TEXT references (four strata / TS / direction / fallback)
    text_checks = (
        ("ts_rule", ("pre_TS", "BOTH")),
        ("path_order_rule", ("endpoint_b_index",)),
        ("direction_rule", ("forward", "reverse")),
        ("fallback_rule", ("fallback_counts",)),
        ("missing_direction_rule", ("SAMPLER_DIRECTION_POOL_EMPTY",)),
    )
    for field, needles in text_checks:
        value = binding.get(field)
        if not isinstance(value, str) or not value.strip():
            raise V541SchemaError(
                "V541_SAMPLER_DRIFT",
                "sampler_binding.{} must state the frozen rule".format(field))
        for needle in needles:
            if needle not in value:
                raise V541SchemaError(
                    "V541_SAMPLER_DRIFT",
                    "sampler_binding.{} must reference {!r} (frozen rule "
                    "text drifted)".format(field, needle),
                )
    refs = binding.get("references")
    if not isinstance(refs, Mapping) or not str(
            refs.get("spec", "")).startswith(
        "pes2ts_core/pesn/offline/training/v541_strata.py#"):
        raise V541SchemaError(
            "V541_SAMPLER_DRIFT",
            "sampler_binding.references.spec must point at "
            "v541_strata.py#SAMPLER_SPEC_V541",
        )


def _check_feature_constants(cfg: Mapping[str, Any]) -> None:
    """Provider constants in the config must equal the LIVE code values."""
    contract = cfg.get("embedding_contract")
    recipe = contract.get("feature_recipe") if isinstance(contract, Mapping) else None
    if not isinstance(recipe, Mapping):
        raise V541SchemaError(
            "V541_MISSING_FIELD", "embedding_contract.feature_recipe missing")
    prov = _code_constants()["prov"]

    def _drift(what: str, got: Any, want: Any) -> None:
        raise V541SchemaError(
            "V541_FEATURE_CONST_DRIFT",
            "feature_recipe.{} = {!r} != live code {!r} (§6.6)".format(
                what, got, want),
        )

    if recipe.get("chem_edge_dim") != prov.CHEM_EDGE_DIM:
        _drift("chem_edge_dim", recipe.get("chem_edge_dim"), prov.CHEM_EDGE_DIM)
    fields = tuple(recipe.get("chem_edge_fields") or ())
    if fields != tuple(prov.CHEM_EDGE_FIELDS):
        _drift("chem_edge_fields", list(fields), list(prov.CHEM_EDGE_FIELDS))
    angle = recipe.get("angle")
    if not isinstance(angle, Mapping):
        raise V541SchemaError(
            "V541_MISSING_FIELD", "feature_recipe.angle missing")
    tie = angle.get("tie_tolerance")
    if not isinstance(tie, (int, float)) or float(tie) != float(
            prov.ANGLE_SHELL_TIE_TOL):
        _drift("angle.tie_tolerance", tie, float(prov.ANGLE_SHELL_TIE_TOL))
    if angle.get("tie_policy") != prov.ANGLE_TIE_POLICY:
        _drift("angle.tie_policy", angle.get("tie_policy"), prov.ANGLE_TIE_POLICY)
    if angle.get("max_neighbors_per_center") != prov.ANGLE_MAX_NEIGHBORS:
        _drift("angle.max_neighbors_per_center",
               angle.get("max_neighbors_per_center"), prov.ANGLE_MAX_NEIGHBORS)
    if angle.get("max_unordered_pairs_per_center") != prov.ANGLE_MAX_PAIRS:
        _drift("angle.max_unordered_pairs_per_center",
               angle.get("max_unordered_pairs_per_center"), prov.ANGLE_MAX_PAIRS)
    rbf = recipe.get("rbf12")
    if not isinstance(rbf, Mapping):
        raise V541SchemaError("V541_MISSING_FIELD", "feature_recipe.rbf12 missing")
    for key, want in (("n_bins", prov.RBF_N),
                      ("center_start_angstrom", prov.RBF_CENTER_START),
                      ("center_end_angstrom", prov.RBF_CENTER_END),
                      ("sigma_angstrom", prov.RBF_SIGMA_ANGSTROM)):
        got = rbf.get(key)
        if isinstance(want, float):
            ok = isinstance(got, (int, float)) and float(got) == float(want)
        else:
            ok = got == want
        if not ok:
            _drift("rbf12.{}".format(key), got, want)

    # --- E2 degree-stats contract (runner-owned, inner-train only) --------
    ds = recipe.get("degree_stats")
    if not isinstance(ds, Mapping):
        raise V541SchemaError(
            "V541_MISSING_FIELD", "feature_recipe.degree_stats missing")
    if ds.get("fit_split") != "inner-train_only":
        raise V541SchemaError(
            "V541_DEGREE_STATS_FIT",
            "degree_stats.fit_split must be 'inner-train_only' — dev / "
            "inner-val / confirm may never fit the E2 degree statistics, got "
            "{!r}".format(ds.get("fit_split")),
        )
    forbidden = ds.get("forbidden_fit_splits")
    if not isinstance(forbidden, Sequence) or isinstance(forbidden, (str, bytes)) \
            or "inner-train_only" in set(str(x) for x in forbidden) \
            or not {"dev", "confirm"} <= set(str(x) for x in forbidden):
        raise V541SchemaError(
            "V541_DEGREE_STATS_FIT",
            "degree_stats.forbidden_fit_splits must name at least dev and "
            "confirm and must not contain 'inner-train_only'",
        )
    required_for = ds.get("required_for")
    if not isinstance(required_for, Sequence) or isinstance(required_for, (str, bytes)) \
            or not required_for:
        raise V541SchemaError(
            "V541_MISSING_FIELD", "degree_stats.required_for must be a non-empty list")
    unknown = sorted({str(x) for x in required_for} - KNOWN_EMBEDDINGS)
    if unknown:
        raise V541SchemaError(
            "V541_EMBEDDING_UNKNOWN",
            "degree_stats.required_for references unknown id(s) {}".format(unknown),
        )
    if ds.get("schema") != "pes2ts_v541_degree_stats_v1":
        raise V541SchemaError(
            "V541_DEGREE_STATS_FIT",
            "degree_stats.schema must be 'pes2ts_v541_degree_stats_v1'",
        )
    # the fit population is the FULL inner-train slice: no sampling, no cap
    if ds.get("population") != "full_inner_train":
        raise V541SchemaError(
            "V541_DEGREE_STATS_FIT",
            "degree_stats.population must be 'full_inner_train' (no sampling, "
            "no cap), got {!r}".format(ds.get("population")),
        )
    if ds.get("max_positions") not in (None, 0):
        raise V541SchemaError(
            "V541_DEGREE_STATS_FIT",
            "degree_stats.max_positions must not be set — the fit uses the "
            "FULL inner-train position set, got {!r}".format(
                ds.get("max_positions")),
        )
    if ds.get("split") != "inner-train":
        raise V541SchemaError(
            "V541_DEGREE_STATS_FIT",
            "degree_stats.split must be 'inner-train' (the only split "
            "v541_degree_stats accepts), got {!r}".format(ds.get("split")),
        )
    if ds.get("seed_source") != "run_seed":
        raise V541SchemaError(
            "V541_DEGREE_STATS_FIT",
            "degree_stats.seed_source must be 'run_seed' (one fit / cache / "
            "SHA per run seed), got {!r}".format(ds.get("seed_source")),
        )
    if ds.get("graph_view") != "G_chem":
        raise V541SchemaError(
            "V541_DEGREE_STATS_FIT",
            "degree_stats.graph_view must be 'G_chem' (chem in-degree only — "
            "no spatial edges), got {!r}".format(ds.get("graph_view")),
        )


def _check_embedding_registry_code(cfg: Mapping[str, Any]) -> None:
    """Config registry rows must mirror the WP2 code registry (§6.1)."""
    contract = cfg.get("embedding_contract")
    registry = contract.get("embedding_registry") if isinstance(contract, Mapping) else None
    if not isinstance(registry, Mapping):
        raise V541SchemaError(
            "V541_MISSING_FIELD", "embedding_contract.embedding_registry missing")
    specs = _code_constants()["reg"].EMBEDDING_SPECS
    for eid, row in registry.items():
        spec = specs.get(eid)
        if spec is None:
            # unknown ids are refused by _check_embedding_registry first
            continue

        def _drift(field: str, got: Any, want: Any) -> None:
            raise V541SchemaError(
                "V541_EMBEDDING_UNKNOWN",
                "embedding_registry.{}.{} = {!r} != WP2 code registry {!r} "
                "(config/code drift)".format(eid, field, got, want),
            )

        for field in ("family", "internal_view", "output"):
            if row.get(field) != getattr(spec, field):
                _drift(field, row.get(field), getattr(spec, field))
        if bool(row.get("in_main_matrix")) != bool(spec.in_main_matrix):
            _drift("in_main_matrix", row.get("in_main_matrix"),
                   spec.in_main_matrix)
        if bool(row.get("diagnostic_only", False)) != bool(spec.diagnostic_only):
            _drift("diagnostic_only", row.get("diagnostic_only"),
                   spec.diagnostic_only)
        if tuple(row.get("conditional_parent") or ()) != tuple(spec.conditional_parent):
            _drift("conditional_parent", row.get("conditional_parent"),
                   list(spec.conditional_parent))


def _check_identity_contract(cfg: Mapping[str, Any]) -> List[str]:
    """Code / common-protocol / environment provenance contract (§12.2/§17).

    Returns extra ``t1_blocked`` reasons that are NOT schema errors (a
    missing R3 checkpoint file blocks T1 but does not make the yaml itself
    invalid — the file is produced/consumed outside git).
    """
    ic = cfg.get("identity_contract")
    if not isinstance(ic, Mapping):
        raise V541SchemaError("V541_MISSING_FIELD", "identity_contract missing")
    blocked: List[str] = []

    code = ic.get("code")
    if not isinstance(code, Mapping):
        raise V541SchemaError("V541_MISSING_FIELD", "identity_contract.code missing")
    if code.get("method") != "source_tree_fingerprint":
        raise V541SchemaError(
            "V541_IDENTITY_CONTRACT",
            "identity_contract.code.method must be 'source_tree_fingerprint' "
            "(the worktree is dirty: a commit alone cannot bind the code)",
        )
    if code.get("git_commit_authoritative") is not False:
        raise V541SchemaError(
            "V541_IDENTITY_CONTRACT",
            "identity_contract.code.git_commit_authoritative must be false",
        )
    git_fn = str(code.get("git_state_fn") or "")
    if git_fn != "pes2ts_core.pesn.offline.training.v541_identity.git_state":
        raise V541SchemaError(
            "V541_IDENTITY_CONTRACT",
            "identity_contract.code.git_state_fn must be the subprocess-free "
            "v541_identity.git_state, got {!r}".format(git_fn),
        )
    includes = code.get("includes")
    if not isinstance(includes, Sequence) or isinstance(includes, (str, bytes)) \
            or not includes:
        raise V541SchemaError(
            "V541_IDENTITY_CONTRACT", "identity_contract.code.includes empty")
    missing = []
    for rel in includes:
        if not (PROJECT_ROOT / str(rel)).exists():
            missing.append(str(rel))
    if missing:
        raise V541SchemaError(
            "V541_IDENTITY_CONTRACT",
            "identity_contract.code.includes entry/entries missing on disk: "
            "{}".format(sorted(missing)),
        )
    suffixes = code.get("suffixes")
    if not isinstance(suffixes, Sequence) or not suffixes:
        raise V541SchemaError(
            "V541_IDENTITY_CONTRACT", "identity_contract.code.suffixes empty")

    common = ic.get("common_protocol")
    if not isinstance(common, Mapping):
        raise V541SchemaError(
            "V541_MISSING_FIELD", "identity_contract.common_protocol missing")
    if common.get("method") != "numeric_protocol_sha256":
        raise V541SchemaError(
            "V541_IDENTITY_CONTRACT",
            "identity_contract.common_protocol.method must be "
            "'numeric_protocol_sha256'",
        )
    excludes = {str(x) for x in (common.get("excludes") or ())}
    includes_c = {str(x) for x in (common.get("includes") or ())}
    must_exclude = {"embedding_id", "feature_recipe_sha256",
                    "embedding_spec_sha256", "seed", "run_id"}
    if not must_exclude <= excludes:
        raise V541SchemaError(
            "V541_IDENTITY_CONTRACT",
            "common_protocol.excludes must keep the per-candidate/per-run "
            "fields OUT of the shared SHA; missing {}".format(
                sorted(must_exclude - excludes)),
        )
    must_include = {"schedule", "lambdas", "batch", "sampler_binding",
                    "split_contract", "backbone_init", "code_sha256",
                    "data_manifest_sha256", "split_sha256", "label_spec_sha256"}
    if not must_include <= includes_c:
        raise V541SchemaError(
            "V541_IDENTITY_CONTRACT",
            "common_protocol.includes must bind the numeric protocol; "
            "missing {}".format(sorted(must_include - includes_c)),
        )
    if excludes & includes_c:
        raise V541SchemaError(
            "V541_IDENTITY_CONTRACT",
            "common_protocol includes/excludes overlap on {}".format(
                sorted(excludes & includes_c)),
        )

    env = ic.get("environment")
    if not isinstance(env, Mapping):
        raise V541SchemaError(
            "V541_MISSING_FIELD", "identity_contract.environment missing")
    if env.get("fn") != "pes2ts_core.pesn.offline.training.v541_identity.env_state":
        raise V541SchemaError(
            "V541_IDENTITY_CONTRACT",
            "identity_contract.environment.fn must be v541_identity.env_state, "
            "got {!r}".format(env.get("fn")),
        )
    if env.get("sha256_field") != "environment_sha256":
        raise V541SchemaError(
            "V541_IDENTITY_CONTRACT",
            "identity_contract.environment.sha256_field must be "
            "'environment_sha256'",
        )
    ident = _code_constants()["ident"]
    if not callable(getattr(ident, "git_state", None)) or not callable(
            getattr(ident, "env_state", None)):
        raise V541SchemaError(
            "V541_IDENTITY_CONTRACT",
            "v541_identity must expose callable git_state/env_state",
        )
    return blocked


def _check_backbone_init(cfg: Mapping[str, Any], *, formal: bool) -> List[str]:
    """R3 init path + SHA (§5.4): present, well-formed, byte-verified."""
    tc = cfg.get("training_contract")
    init = tc.get("backbone_init") if isinstance(tc, Mapping) else None
    if not isinstance(init, Mapping):
        raise V541SchemaError(
            "V541_MISSING_FIELD", "training_contract.backbone_init missing")
    path = init.get("path")
    sha = init.get("sha256")
    if not isinstance(path, str) or not path.strip():
        raise V541SchemaError(
            "V541_R3_INIT", "backbone_init.path must be a non-empty path")
    if not isinstance(sha, str) or not _SHA64_RE.match(sha) or sha == "0" * 64:
        raise V541SchemaError(
            "V541_R3_INIT",
            "backbone_init.sha256 must be a non-zero 64-hex digest, got "
            "{!r}".format(sha),
        )
    if init.get("verify_on_validate") is not True:
        raise V541SchemaError(
            "V541_R3_INIT", "backbone_init.verify_on_validate must be true")
    blocked: List[str] = []
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    if not target.is_file():
        if formal:
            raise V541SchemaError(
                "V541_R3_INIT_MISSING",
                "R3 init checkpoint {} is missing — a formal run cannot "
                "start from an unverifiable init".format(path),
            )
        blocked.append(
            "backbone_init file missing: {} (R3 init SHA cannot be "
            "re-verified until the checkpoint is present)".format(path))
        return blocked
    actual = _sha256_of_file(target)
    if actual != sha:
        raise V541SchemaError(
            "V541_R3_INIT_DRIFT",
            "R3 init {} hashes to {} but config carries {} — restore the "
            "checkpoint or re-sign the config".format(
                path, actual[:12], sha[:12]),
        )
    return blocked


def _check_t3_preregistration(cfg: Mapping[str, Any], runs: Sequence[Any]) -> None:
    """T3 must preregister ALL selectable candidates (plan §9.1/§9.5).

    Cross product E1–E5 × seeds {29,43} (10 stable ``V541_E#_S##`` ids) is
    listed as conditional runs; only the inner-val top-2 actually activates
    (max 4 runs).  Unactivated entries record ``gate_skipped`` and add 0
    trainings; ``N_core`` follows the §9.1 formula, never a sum of listed
    runs.  E5 rows must stack the parent gate on top of the T3 selection gate.
    """
    rm = cfg.get("run_matrix")
    block = rm.get("t3_preregistration")
    if not isinstance(block, Mapping):
        raise V541SchemaError(
            "V541_RUN_MATRIX", "run_matrix.t3_preregistration block missing"
        )
    if tuple(block.get("candidates") or ()) != T3_CANDIDATES:
        raise V541SchemaError(
            "V541_RUN_MATRIX",
            f"t3_preregistration.candidates must be {list(T3_CANDIDATES)}, "
            f"got {block.get('candidates')!r}",
        )
    if tuple(block.get("seeds") or ()) != T3_SEEDS:
        raise V541SchemaError(
            "V541_RUN_MATRIX",
            f"t3_preregistration.seeds must be {list(T3_SEEDS)}, "
            f"got {block.get('seeds')!r}",
        )
    expected_count = len(T3_CANDIDATES) * len(T3_SEEDS)  # 10
    if block.get("listed_runs") != expected_count:
        raise V541SchemaError(
            "V541_RUN_MATRIX",
            f"t3_preregistration.listed_runs must be {expected_count}, "
            f"got {block.get('listed_runs')!r}",
        )
    if block.get("full_activation_count") != T3_FULL_ACTIVATION:
        raise V541SchemaError(
            "V541_RUN_MATRIX",
            f"t3_preregistration.full_activation_count must be "
            f"{T3_FULL_ACTIVATION} (2 frozen candidates x 2 seeds, §9.1), "
            f"got {block.get('full_activation_count')!r}",
        )
    if block.get("activation") != T3_ACTIVATION:
        raise V541SchemaError(
            "V541_RUN_MATRIX",
            f"t3_preregistration.activation must be {T3_ACTIVATION!r}, "
            f"got {block.get('activation')!r}",
        )
    if block.get("unactivated_status") != T3_UNACTIVATED_STATUS:
        raise V541SchemaError(
            "V541_RUN_MATRIX",
            f"t3_preregistration.unactivated_status must be "
            f"{T3_UNACTIVATED_STATUS!r}, got {block.get('unactivated_status')!r}",
        )
    if block.get("n_core_counting") != "formula_only":
        raise V541SchemaError(
            "V541_RUN_MATRIX",
            "t3_preregistration.n_core_counting must be 'formula_only' — "
            "N_core is the §9.1 formula, never a sum of listed runs",
        )

    t3_runs = [r for r in runs if isinstance(r, Mapping) and r.get("stage") == "T3"]
    expected_ids = {
        f"V541_{eid}_S{seed}" for eid in T3_CANDIDATES for seed in T3_SEEDS
    }
    actual_ids = {r.get("run_id") for r in t3_runs}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(x for x in actual_ids - expected_ids if x is not None)
        raise V541SchemaError(
            "V541_RUN_MATRIX",
            f"T3 preregistration must list exactly {len(expected_ids)} runs "
            f"(E1–E5 x seeds {list(T3_SEEDS)}); missing={missing} extra={extra}",
        )
    if len(t3_runs) != expected_count:
        raise V541SchemaError(
            "V541_RUN_MATRIX", "duplicate/overlapping T3 run entries detected"
        )
    for run in t3_runs:
        run_id = str(run.get("run_id"))
        if T3_RUN_ID_TOKEN in run_id:
            raise V541SchemaError(
                "V541_RUN_MATRIX",
                f"{run_id}: ambiguous TOP2 placeholder ids are forbidden; every "
                "T3 run carries a deterministic embedding identity",
            )
        if run.get("class") != "conditional":
            raise V541SchemaError(
                "V541_RUN_MATRIX", f"{run_id}: T3 runs must be class=conditional"
            )
        if run.get("winner_eligible") is not True:
            raise V541SchemaError(
                "V541_RUN_MATRIX", f"{run_id}: T3 runs must be winner_eligible=true"
            )
        if run.get("condition") != T3_ACTIVATION:
            raise V541SchemaError(
                "V541_RUN_MATRIX",
                f"{run_id}: condition must be {T3_ACTIVATION!r} "
                f"(only inner-val top-2 activates), got {run.get('condition')!r}",
            )
        if run.get("unactivated_status") != T3_UNACTIVATED_STATUS:
            raise V541SchemaError(
                "V541_RUN_MATRIX",
                f"{run_id}: unactivated_status must be {T3_UNACTIVATED_STATUS!r}",
            )
        if run.get("new_training_if_activated") != 1:
            raise V541SchemaError(
                "V541_RUN_MATRIX",
                f"{run_id}: must declare new_training_if_activated=1 (a "
                "not-activated T3 run contributes 0 new trainings)",
            )
        if run.get("new_training") is not None:
            raise V541SchemaError(
                "V541_RUN_MATRIX",
                f"{run_id}: T3 entries must not carry an unconditional "
                "'new_training' field — use new_training_if_activated only",
            )
        eid = run.get("embedding")
        if run.get("embedding") is None:
            raise V541SchemaError(
                "V541_EMBEDDING_UNKNOWN",
                f"{run_id}: run lacks a deterministic embedding identity",
            )
        if eid == "E5":
            if run.get("parent_gate") != T3_PARENT_GATE:
                raise V541SchemaError(
                    "V541_RUN_MATRIX",
                    f"{run_id}: E5 T3 run must stack parent_gate="
                    f"{T3_PARENT_GATE!r} on top of the T3 selection gate",
                )
        elif "parent_gate" in run:
            raise V541SchemaError(
                "V541_RUN_MATRIX",
                f"{run_id}: parent_gate is only valid on E5 T3 rows "
                "(E1–E4 are unconditional parents)",
            )


def _check_run_matrix(cfg: Mapping[str, Any]) -> List[str]:
    """Validate run matrix shape; return ids that are winner-eligible."""
    rm = cfg.get("run_matrix")
    if not isinstance(rm, Mapping):
        raise V541SchemaError("V541_MISSING_FIELD", "run_matrix must be a mapping")
    classes = rm.get("classes")
    if not isinstance(classes, Sequence) or set(classes) != RUN_CLASSES:
        raise V541SchemaError(
            "V541_RUN_MATRIX",
            f"run_matrix.classes must be exactly {sorted(RUN_CLASSES)}, got {classes!r}",
        )
    runs = rm.get("runs")
    if not isinstance(runs, Sequence) or not runs:
        raise V541SchemaError("V541_MISSING_FIELD", "run_matrix.runs missing/empty")
    seeds = set(cfg.get("split_contract", {}).get("seeds") or [])
    winners: List[str] = []
    seen_ids: set[str] = set()
    for entry in runs:
        if not isinstance(entry, Mapping):
            raise V541SchemaError("V541_RUN_MATRIX", f"run entry not a mapping: {entry!r}")
        run_id = entry.get("run_id")
        if not isinstance(run_id, str) or not run_id.startswith("V541_"):
            raise V541SchemaError(
                "V541_RUN_MATRIX",
                f"run_id {run_id!r} must use the V541_ namespace (D05)",
            )
        if T3_RUN_ID_TOKEN in run_id:
            raise V541SchemaError(
                "V541_RUN_MATRIX",
                f"{run_id}: ambiguous TOP2 placeholder ids are forbidden; every "
                "T3 run must carry a deterministic V541_E#_S## identity",
            )
        if run_id in seen_ids:
            raise V541SchemaError("V541_RUN_MATRIX", f"duplicate run_id {run_id!r}")
        seen_ids.add(run_id)
        if entry.get("class") not in RUN_CLASSES:
            raise V541SchemaError(
                "V541_RUN_MATRIX",
                f"{run_id}: class {entry.get('class')!r} not in {sorted(RUN_CLASSES)}",
            )
        if seeds and entry.get("seed") not in seeds:
            raise V541SchemaError(
                "V541_RUN_MATRIX",
                f"{run_id}: seed {entry.get('seed')!r} not in split seeds {sorted(seeds)}",
            )
        eid = entry.get("embedding")
        if eid is None:
            raise V541SchemaError(
                "V541_EMBEDDING_UNKNOWN",
                f"{run_id}: run lacks a deterministic embedding identity "
                "(every preregistered run_id binds one frozen embedding id)",
            )
        if eid not in KNOWN_EMBEDDINGS:
            raise V541SchemaError(
                "V541_EMBEDDING_UNKNOWN",
                f"{run_id}: embedding {eid!r} outside frozen E0–E6 registry",
            )
        if eid == "E6":
            raise V541SchemaError(
                "V541_EMBEDDING_UNKNOWN",
                f"{run_id}: E6 is diagnostic-only and may not be scheduled in the "
                "run matrix without separate approval (§6.1)",
            )
        if bool(entry.get("winner_eligible")):
            if entry.get("class") == "diagnostic":
                raise V541SchemaError(
                    "V541_RUN_MATRIX",
                    f"{run_id}: diagnostic runs are never winner-eligible (§9.1)",
                )
            winners.append(run_id)
    if not winners:
        raise V541SchemaError(
            "V541_RUN_MATRIX", "no winner-eligible (nominal/conditional) run registered"
        )
    return winners


def _check_gv_block(cfg: Mapping[str, Any]) -> None:
    gv = cfg.get("gv")
    if not isinstance(gv, Mapping):
        raise V541SchemaError("V541_MISSING_FIELD", "gv must be a mapping")
    missing = [k for k in _GV_DEFAULT if gv.get(k) is None]
    if missing:
        raise V541SchemaError("V541_MISSING_FIELD", f"gv missing field(s): {missing}")
    try:
        gv_map = load_gv_map(DEFAULT_MAP_PATH)
        campaign = resolve_campaign(gv_map, str(gv["campaign_id"]))
    except GVMapError as exc:
        raise V541SchemaError("V541_GV_MISMATCH", str(exc)) from exc
    if gv["campaign_id"] != GV_CAMPAIGN_ID:
        raise V541SchemaError(
            "V541_GV_MISMATCH",
            f"gv.campaign_id must be {GV_CAMPAIGN_ID!r}, got {gv['campaign_id']!r}",
        )
    if gv.get("primary_g") != campaign.primary_g:
        raise V541SchemaError(
            "V541_GV_MISMATCH",
            f"gv.primary_g {gv.get('primary_g')!r} != registry {campaign.primary_g!r}",
        )
    if set(gv.get("supporting_g") or ()) != set(campaign.supporting_g):
        raise V541SchemaError(
            "V541_GV_MISMATCH",
            f"gv.supporting_g {sorted(gv.get('supporting_g') or ())} != registry "
            f"{sorted(campaign.supporting_g)}",
        )
    if set(gv.get("gates") or ()) != set(campaign.gates):
        raise V541SchemaError(
            "V541_GV_MISMATCH",
            f"gv.gates {sorted(gv.get('gates') or ())} != registry "
            f"{sorted(campaign.gates)}",
        )
    if gv.get("map_version") != gv_map.map_version:
        raise V541SchemaError(
            "V541_GV_MISMATCH",
            f"gv.map_version {gv.get('map_version')!r} != {gv_map.map_version!r}",
        )


def _check_thresholds(
    cfg: Mapping[str, Any], *, formal: bool
) -> List[str]:
    th = cfg.get("thresholds")
    if not isinstance(th, Mapping):
        raise V541SchemaError("V541_MISSING_FIELD", "thresholds must be a mapping")
    nulls = [k for k in FORMAL_THRESHOLDS if th.get(k) is None]
    if formal:
        if nulls:
            raise V541SchemaError(
                "V541_THRESHOLD_NULL",
                f"formal gate: threshold(s) still null {nulls}; unsigned values "
                "block T1 (runner must not fill defaults, §3.3/§11.2)",
            )
        signoff = th.get("signoff")
        if not isinstance(signoff, Mapping) or signoff.get("status") != "frozen":
            raise V541SchemaError(
                "V541_THRESHOLD_NULL",
                "formal gate: thresholds.signoff.status must be 'frozen' before T1",
            )
        for key in FORMAL_THRESHOLDS:
            if signoff.get(key) != th.get(key):
                raise V541SchemaError(
                    "V541_THRESHOLD_NULL",
                    f"formal gate: thresholds.signoff.{key} "
                    f"({signoff.get(key)!r}) != thresholds.{key} ({th.get(key)!r}); "
                    "signed record must mirror the config values byte-for-byte",
                )
    for key in FORMAL_THRESHOLDS:
        if key in th and th[key] is not None and not isinstance(
            th[key], (int, float)
        ):
            raise V541SchemaError(
                "V541_THRESHOLD_NULL",
                f"threshold {key!r} must be numeric, got {th[key]!r}",
            )
    return nulls


def _check_run_outcomes(cfg: Mapping[str, Any]) -> None:
    ro = cfg.get("run_outcomes")
    if not isinstance(ro, Mapping):
        raise V541SchemaError("V541_MISSING_FIELD", "run_outcomes must be a mapping")
    vocab = ro.get("vocabulary")
    expected = {
        "done",
        "invalid_run",
        "scientific_failure",
        "resource_failure",
        "budget_censored",
        "gate_skipped",
    }
    if not isinstance(vocab, Sequence) or set(vocab) != expected:
        raise V541SchemaError(
            "V541_RUN_MATRIX",
            f"run_outcomes.vocabulary must be exactly {sorted(expected)} (§7.4), "
            f"got {vocab!r}",
        )
    if ro.get("ledger_status_machine") != "unchanged":
        raise V541SchemaError(
            "V541_RUN_MATRIX",
            "run_outcomes.ledger_status_machine must be 'unchanged' — V541 run "
            "outcome is tracked separately from the legacy ledger status machine",
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

    Raises :class:`V541SchemaError` on the first typed rejection.  The report
    carries ``mode``, ``checks`` (passed check ids), ``null_thresholds`` and
    ``t1_blocked`` reasons (empty only when nothing blocks T1).
    """
    checks: List[str] = []

    _check_schema(cfg)
    checks.append("schema")
    # objective precedes the generic required-field sweep so a null/empty
    # objective reports V541_OBJECTIVE (the semantic error) not MISSING_FIELD.
    _check_objective_and_arms(cfg)
    checks.append("objective")
    _check_required_fields(cfg)
    checks.append("required_fields")
    _check_embedding_registry(cfg)
    checks.append("embedding_registry")
    _check_forward_view(cfg)
    checks.append("forward_view")
    _check_core_consumption(cfg)
    checks.append("core_consumption")
    _check_sampler_binding(cfg)
    checks.append("sampler_binding")
    _check_feature_constants(cfg)
    checks.append("feature_constants")
    _check_embedding_registry_code(cfg)
    checks.append("embedding_registry_code")
    identity_blocked = _check_identity_contract(cfg)
    checks.append("identity_contract")
    r3_blocked = _check_backbone_init(cfg, formal=formal)
    checks.append("backbone_init")
    _check_authorization(cfg)
    checks.append("authorization")
    if check_gv:
        _check_gv_block(cfg)
        checks.append("gv_registry")
    _check_run_matrix(cfg)
    checks.append("run_matrix")
    _check_t3_preregistration(cfg, cfg.get("run_matrix").get("runs"))
    checks.append("t3_preregistration")
    _check_run_outcomes(cfg)
    checks.append("run_outcomes")
    null_thresholds = _check_thresholds(cfg, formal=formal)
    checks.append("thresholds")

    t1_blocked: List[str] = []
    t1_blocked.extend(identity_blocked)
    t1_blocked.extend(r3_blocked)
    if null_thresholds:
        t1_blocked.append(
            f"unsigned threshold(s) {null_thresholds} block T1 until signed (§11.2)"
        )
    budget = cfg.get("budget")
    if isinstance(budget, Mapping) and budget.get("per_candidate_core_time") is None:
        t1_blocked.append(
            "budget.per_candidate_core_time null: W0/W1 profiling signoff "
            "required before formal T1 (§8.3/§17)"
        )
    auth = cfg.get("governance", {}).get("authorization", {})
    for gate in auth.get("t1_start_requires", []) if isinstance(auth, Mapping) else []:
        pass  # gate booleans are runner-side; config only declares the list

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
        help="formal gate: require non-null signed thresholds (blocks T1 otherwise)",
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
    except V541SchemaError as exc:
        if args.json:
            print(json.dumps({"ok": False, "code": exc.code, "detail": exc.detail}))
        else:
            print(f"V541 config rejected: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            "V541 config OK: schema={schema} objective={objective} mode={mode} "
            "checks={n}".format(n=len(report["checks"]), **report)
        )
        if report["t1_blocked"]:
            print("T1 not config-ready:")
            for reason in report["t1_blocked"]:
                print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
