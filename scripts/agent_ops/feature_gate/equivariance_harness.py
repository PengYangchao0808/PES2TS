"""Feature-gate property-test harness (T1.1-T1.4 + T1.4b/T1.4c), v2.

Model-agnostic: every ``check_*`` takes a ``model`` and a feature ``batch``
(a plain dict of tensors / metadata) and returns a **report dict**; nothing
here mutates the caller's batch.

v2 (WP-F2, Oracle ORA-2 blockers) changed the preset regime from "warn and
pick" to **fail-closed normative binding**:

* the normative preset is ``30h_v1.1_normative`` (rel 1e-5 + abs 1e-6 for
  rotation/translation/permutation, relabel max abs 1e-6), bound to
  ``config/campaigns/pes2ts_30h_20261020.yaml`` ``feature_contract.tolerances``
  (PD-2/PD-6).  Tolerances may be read from that yaml via
  :func:`load_campaign_feature_contract` +
  :func:`verify_preset_against_yaml`; a yaml/preset mismatch is a hard
  :class:`PresetContractError` ("禁止自动放宽" — no automatic relaxation).
* the old ``r3_v3`` (abs 1e-4) value survives **only** as the record-only
  preset ``r3_v3_record``: reports generated with it carry
  ``gate_admissible=false`` and the banner "record-only, not a pass gate".
* ``preset=None`` is **refused** in strict mode (the default).  Legacy
  behaviour is an explicit opt-in (``strict=False``) for diagnostics only;
  such reports carry ``preset_explicit=false`` -> ``gate_admissible=false``
  -> overall ``passed=false``.  There is no warning-and-pass path.
* harness version constant: :data:`HARNESS_VERSION` == ``"2"``.

Model contract (unchanged from v1, minimal so StarEGNN v3 can be adapted):

* input  — ``batch`` dict; coordinate-bearing tensors are float ``[*, 3]``;
  the harness only requires ``"pos"`` ``[N, 3]`` for geometric transforms.
  Optional conventional keys: ``"z"`` ``[N]``, ``"edge_index"`` ``[2, E]``,
  ``"key_index"`` ``[K]``, ``"graph_index"`` ``[N]``, ``"endpoint_names"``
  (tuple, relabel check), ``"signed_intent"`` (scalar, reverse check) plus
  the reaction-intent fields exercised by the intent-off mask check.
* output — ``model(batch)`` returns a tensor ``u [K, 3]`` or a dict with
  ``"u"`` (equivariant vectors) plus optional invariant scalar tensors.

Normative references (all committed on master at ``29f0e25``):

* ``config/campaigns/pes2ts_30h_20261020.yaml`` ``feature_contract`` —
  tolerances, intent_off_mask five classes, forbidden_model_inputs (PD-2/PD-6).
* ``docs/contracts/PES2TS_ReactionIntent_Contract_v1.md`` L321-343 — forward
  input legality (allowed/forbidden classes).
* ``docs/contracts/PES2TS_GeometryPolicyContract_v1.md`` S4.3
  (L121/L125/L170/L183) — NaN/out-of-contract input -> typed invalid, never
  a fabricated action.
* ``docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md``
  S7.4 table T1.1-T1.4 (gate states ``pass|fail|blocked|partial|not_run``).
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

__all__ = [
    "HARNESS_VERSION",
    "PRESETS",
    "PRESET_SOURCES",
    "DEFAULT_PRESET",
    "DEFAULT_CAMPAIGN_YAML",
    "RECORD_ONLY_BANNER",
    "RELABEL_ABS_TOL",
    "FORBIDDEN_FEATURE_PATTERNS",
    "FORWARD_INPUT_ALLOWLIST",
    "FORWARD_INPUT_ALLOWLIST_GROUPS",
    "FORWARD_INPUT_ALLOWLIST_SOURCE",
    "INTENT_OFF_MASK_CLASSES",
    "PRESERVED_UNORDERED_CONTEXT_KEYS",
    "PresetRequiredError",
    "PresetContractError",
    "IllegalFeatureError",
    "TypedInvalidError",
    "resolve_preset",
    "load_campaign_feature_contract",
    "verify_preset_against_yaml",
    "check_input_legality",
    "check_input_allowlist",
    "check_intent_off_mask",
    "check_translation_invariance",
    "check_rotation_equivariance",
    "check_permutation_equivariance",
    "check_relabel_invariance",
    "check_reverse_sensitivity",
    "check_nan_typed_invalid",
    "run_feature_gates",
    "build_gate_report",
]

#: harness v2 (WP-F2).  Bump on any report-schema / preset-semantics change.
HARNESS_VERSION = "2"

#: campaign yaml that the normative tolerances are bound to (PD-2/PD-6).
DEFAULT_CAMPAIGN_YAML = "config/campaigns/pes2ts_30h_20261020.yaml"

#: banner carried by every report generated with a record-only preset.
RECORD_ONLY_BANNER = "record-only, not a pass gate"


# ---------------------------------------------------------------------------
# Tolerance presets — normative binding (v2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Preset:
    name: str
    rel: float
    abs: float  # noqa: A003 - mirrors numpy/pytorch naming
    source: str
    #: False for record-only reference values: such reports are diagnostics,
    #: never a pass gate (ORA-2).
    gate_admissible: bool = True
    record_only: bool = False
    #: hard bit-identity threshold for the endpoint display-name swap
    #: (feature_contract.tolerances.relabel_invariance_max_abs, PD-2).
    relabel_max_abs: float = 1e-6
    banner: str | None = None


PRESET_SOURCES: dict[str, str] = {
    "30h_v1.1_normative": (
        "config/campaigns/pes2ts_30h_20261020.yaml feature_contract."
        "tolerances (rotation_translation_permutation rel 1e-5/abs 1e-6, "
        "relabel_invariance_max_abs 1e-6; PD-2/PD-6), commit 29f0e25; "
        "docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md "
        "S3.3/S7.4 T1.1-T1.3"
    ),
    "r3_v3_record": (
        "config/campaigns/pes2ts_30h_20261020.yaml feature_contract."
        "tolerances.record_only_reference 1.0e-4 (r3_v3 old value, 仅记录性对照), "
        "commit 29f0e25 — RECORD-ONLY, not a pass gate"
    ),
}

PRESETS: dict[str, Preset] = {
    "30h_v1.1_normative": Preset(
        name="30h_v1.1_normative", rel=1e-5, abs=1e-6,
        source=PRESET_SOURCES["30h_v1.1_normative"],
        gate_admissible=True, record_only=False, relabel_max_abs=1e-6,
    ),
    "r3_v3_record": Preset(
        name="r3_v3_record", rel=0.0, abs=1e-4,
        source=PRESET_SOURCES["r3_v3_record"],
        gate_admissible=False, record_only=True, relabel_max_abs=1e-6,
        banner=RECORD_ONLY_BANNER,
    ),
}

DEFAULT_PRESET = "30h_v1.1_normative"

#: legacy (v1) names — rejected in v2 with a rename hint.
_LEGACY_PRESET_NAMES = ("30h_v1.1", "r3_v3")

#: Hard bit-identity / abs threshold for the endpoint display-name swap.
#: Kept as a module constant for callers, but the authoritative value lives in
#: ``feature_contract.tolerances.relabel_invariance_max_abs`` (yaml, PD-2).
RELABEL_ABS_TOL = 1e-6


class PresetRequiredError(ValueError):
    """v2 fail-closed refusal: no preset given in strict mode."""


class PresetContractError(ValueError):
    """Preset values disagree with the campaign yaml feature_contract."""


class TypedInvalidError(RuntimeError):
    """Typed invalid raised by models for out-of-contract input (NaN etc.).

    Mirrors GeometryPolicyContract v1 S4.3: never fabricate an action.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class IllegalFeatureError(ValueError):
    """Raised when the feature dict carries forbidden (label-leaking) keys."""

    def __init__(self, violations: Sequence[str]) -> None:
        self.violations = tuple(violations)
        super().__init__(
            "forbidden feature keys (label leakage): "
            + ", ".join(self.violations)
        )


def resolve_preset(preset: str | None, *, strict: bool = True) -> tuple[Preset, bool]:
    """Resolve a preset name.

    v2 semantics (fail-closed):

    * ``preset=None`` + ``strict=True`` (default) -> :class:`PresetRequiredError`
      — no warning-and-pass path exists.
    * ``preset=None`` + ``strict=False`` -> legacy diagnostic fallback: a
      :class:`UserWarning` is emitted and the returned ``explicit=False`` makes
      every report ``gate_admissible=false`` / overall ``passed=false``.
    * an explicit name -> ``explicit=True``; admissibility then depends on the
      preset itself (``r3_v3_record`` is never admissible).
    """
    if preset is None:
        if strict:
            raise PresetRequiredError(
                "preset is required in strict mode (harness v2): pass "
                "preset='30h_v1.1_normative' (normative, bound to "
                f"{DEFAULT_CAMPAIGN_YAML} feature_contract.tolerances) or "
                "preset='r3_v3_record' (record-only diagnostics). Legacy "
                "fallback requires explicit strict=False and yields "
                "gate_admissible=false."
            )
        warnings.warn(
            "feature-gate tolerance preset not specified; using legacy "
            f"fallback '{DEFAULT_PRESET}' with gate_admissible=false "
            "(diagnostics only, never a pass gate). Candidates: "
            + "; ".join(
                f"{name} rel={p.rel} abs={p.abs}"
                + (" [record-only]" if p.record_only else " [normative]")
                for name, p in PRESETS.items()
            ),
            UserWarning,
            stacklevel=2,
        )
        return PRESETS[DEFAULT_PRESET], False
    if preset in _LEGACY_PRESET_NAMES:
        raise KeyError(
            f"legacy preset name {preset!r} was retired in harness v2; use "
            f"'{DEFAULT_PRESET}' (normative) or 'r3_v3_record' "
            f"(record-only). Known: {sorted(PRESETS)}"
        )
    if preset not in PRESETS:
        raise KeyError(
            f"unknown tolerance preset {preset!r}; known: {sorted(PRESETS)}"
        )
    return PRESETS[preset], True


def _admissible(preset: Preset, explicit: bool) -> bool:
    return bool(preset.gate_admissible and explicit)


# ---------------------------------------------------------------------------
# Campaign yaml binding (PD-2: tolerances are normative there)
# ---------------------------------------------------------------------------

def _extract_top_level_block(text: str, key: str) -> str:
    """Extract a top-level ``key:`` block (plus its indented body).

    The 30h campaign yaml is a human-edited DRAFT whose *other* sections are
    not guaranteed to be whole-file parseable (e.g. L96 ``heads:`` uses a flow
    mapping PyYAML rejects).  Feature-gate tolerance binding must not depend
    on the validity of unrelated sections, so the loader falls back to a
    targeted, line-oriented extraction of the ``feature_contract`` block.
    """
    lines = text.splitlines()
    prefix = f"{key}:"
    start = None
    for i, line in enumerate(lines):
        if line.startswith(prefix) and not line.startswith(f"{key}_:"):
            start = i
            break
    if start is None:
        raise PresetContractError(
            f"FEATURE_CONTRACT_MISSING: no top-level {key!r} block")
    body: list[str] = []
    for line in lines[start + 1:]:
        if not line.strip() or line.lstrip().startswith("#"):
            body.append(line)
            continue
        if not line[0].isspace():
            break  # next top-level key
        body.append(line)
    return "\n".join(lines[start:start + 1] + body)


def load_campaign_feature_contract(
    campaign_yaml: str | Path = DEFAULT_CAMPAIGN_YAML,
) -> tuple[dict[str, Any], str]:
    """Read ``feature_contract`` from the campaign yaml; return it + sha256.

    Returns ``(feature_contract_dict, sha256_of_full_file)``.  The sha256
    always covers the **whole file** (that is what the G1 consumer diffs);
    parsing prefers the whole document and falls back to a targeted
    ``feature_contract`` block extraction when other sections are malformed.
    """
    import yaml  # local import: PyYAML is only needed for yaml-backed runs

    path = Path(campaign_yaml)
    if not path.is_file():
        raise PresetContractError(
            f"CAMPAIGN_YAML_MISSING: {path} (feature_contract.tolerances is "
            "the normative tolerance source, PD-2)"
        )
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")

    doc: Any = None
    parse_mode = "whole_file"
    whole_file_error: str | None = None
    try:
        doc = yaml.safe_load(text)
    except Exception as exc:  # noqa: BLE001 - fail-closed with fallback
        whole_file_error = str(exc).splitlines()[0]
        parse_mode = "block_extract"
        try:
            block = _extract_top_level_block(text, "feature_contract")
            doc = yaml.safe_load(block)
        except PresetContractError:
            raise
        except Exception as exc2:  # noqa: BLE001
            raise PresetContractError(
                f"CAMPAIGN_YAML_UNPARSEABLE: {path}: whole-file parse failed "
                f"({whole_file_error}) and feature_contract block extraction "
                f"failed too: {exc2}"
            ) from exc2

    if not isinstance(doc, dict):
        raise PresetContractError(
            f"CAMPAIGN_YAML_INVALID: {path} is not a mapping")
    contract = doc.get("feature_contract")
    if not isinstance(contract, dict) or "tolerances" not in contract:
        raise PresetContractError(
            f"FEATURE_CONTRACT_MISSING: {path} has no feature_contract."
            "tolerances block"
        )
    contract = dict(contract)
    contract["_parse_mode"] = parse_mode
    if whole_file_error:
        contract["_whole_file_parse_error"] = whole_file_error
    return contract, sha


def verify_preset_against_yaml(
    preset: Preset,
    feature_contract: Mapping[str, Any],
    yaml_sha256: str,
) -> dict[str, Any]:
    """Cross-check a preset against ``feature_contract.tolerances``.

    Any disagreement is a :class:`PresetContractError` — tolerances are
    frozen (PD-2 "禁止自动放宽": no automatic relaxation in either direction).
    Returns the yaml-side tolerance block for report provenance.
    """
    tol = feature_contract.get("tolerances")
    if not isinstance(tol, Mapping):
        raise PresetContractError("feature_contract.tolerances is not a mapping")
    try:
        relabel = float(tol["relabel_invariance_max_abs"])
        rtp = tol["rotation_translation_permutation"]
        rel = float(rtp["rel"])
        abs_tol = float(rtp["abs"])  # type: ignore[index]
        record = float(tol["record_only_reference"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PresetContractError(
            f"FEATURE_CONTRACT_TOLERANCES_MALFORMED: {exc}"
        ) from exc

    problems: list[str] = []
    if preset.record_only:
        if preset.abs != record:
            problems.append(
                f"record-only abs {preset.abs} != yaml "
                f"record_only_reference {record}")
    else:
        if preset.rel != rel or preset.abs != abs_tol:
            problems.append(
                f"normative rel/abs {preset.rel}/{preset.abs} != yaml "
                f"rotation_translation_permutation {rel}/{abs_tol}")
        if preset.relabel_max_abs != relabel:
            problems.append(
                f"relabel max abs {preset.relabel_max_abs} != yaml "
                f"relabel_invariance_max_abs {relabel}")
    if problems:
        raise PresetContractError(
            "PRESET_YAML_MISMATCH (frozen contract, no auto-relaxation): "
            + "; ".join(problems)
            + f" [yaml sha256 {yaml_sha256}]"
        )
    return {
        "relabel_invariance_max_abs": relabel,
        "rotation_translation_permutation": {"rel": rel, "abs": abs_tol},
        "record_only_reference": record,
    }


# ---------------------------------------------------------------------------
# Blacklist: forbidden input classes (ReactionIntent Contract v1 L321-343)
# ---------------------------------------------------------------------------

#: Substring patterns (lower-cased) that must never appear in a feature dict.
#: Categories mirror the yaml ``feature_contract.forbidden_model_inputs`` and
#: contract L327-343.  Extend via the ``extra_patterns`` argument of
#: :func:`check_input_legality` (the tuple itself stays frozen).
FORBIDDEN_FEATURE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("future_frames", (
        "future", "next_frame", "pos_next", "frame_next", "fut_",
        "irc_next", "traj_future", "past_frame", "prev_frame", "frame_prev",
    )),
    ("ts_coords", (
        "pos_ts", "ts_coord", "ts_xyz", "x_ts", "true_ts", "ts_pos",
        "ts_frame",
    )),
    ("true_progress_arc", (
        "true_progress", "progress", "remaining_arc", "arc_remaining",
        "s_remaining", "frac_remaining", "true_arc", "arc_length",
        "true_s",
    )),
    ("branch_id", ("branch_id", "branch")),
    ("split_family", (
        "split", "family", "fold", "cluster_id", "group_ids",
        "reaction_ids", "scaffold", "conflict",
    )),
    ("energy_gradient", (
        "energy", "grad", "force", "dft_", "e_react", "e_prod", "e_t",
        "delta_e",
    )),
    ("near_ts_capped_flags", (
        "near_ts", "capped", "cap_flag", "terminal", "d_eff", "deff",
    )),
    # --- v2 additions (ORA-2 item 2: forbidden input classes complete) ------
    ("history_hidden_state", (
        "history", "hidden", "hidden_state", "state_hist", "h_prev",
        "prev_h", "rnn_state", "gru", "lstm", "memory", "hidden_states",
        "irc_history", "cached_state",
    )),
    ("endpoint_geometry_XA_XB", (
        "x_a", "x_b", "xa_", "xb_", "pos_a", "pos_b", "geom_a", "geom_b",
        "endpoint_geom", "endpoint_xyz", "endpoint_coord", "endpoint_pos",
        "x_endpoint",
    )),
    ("endpoint_derived_dA_dB_s", (
        "d_a_", "_d_a", "d_b_", "_d_b", "d_arc", "s_arc", "arc_s",
        "s_to_endpoint", "d_endpoint", "endpoint_d", "path_s",
    )),
    ("travel_direction", (
        "travel_direction", "travel_dir", "direction_token", "dir_token",
        "direction_src", "direction_tgt", "forward_flag", "backward_flag",
        "is_reverse", "reverse_flag",
    )),
)


def _violating_keys(
    batch: Mapping[str, Any],
    extra_patterns: Sequence[tuple[str, tuple[str, ...]]] = (),
) -> list[tuple[str, str]]:
    classes = tuple(FORBIDDEN_FEATURE_PATTERNS) + tuple(extra_patterns)
    hits: list[tuple[str, str]] = []
    for key in batch:
        low = str(key).lower()
        for category, patterns in classes:
            if any(pat in low for pat in patterns):
                hits.append((str(key), category))
                break
    return hits


def check_input_legality(
    batch: Mapping[str, Any],
    *,
    raise_on_violation: bool = False,
    extra_patterns: Sequence[tuple[str, tuple[str, ...]]] = (),
    preset: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Blacklist gate: the feature dict must carry no forbidden keys."""
    preset_obj, explicit = resolve_preset(preset, strict=strict)
    hits = _violating_keys(batch, extra_patterns)
    report: dict[str, Any] = {
        "check": "input_legality",
        "enforcement": "blacklist",
        "passed": not hits,
        "violations": [k for k, _ in hits],
        "violation_categories": sorted({c for _, c in hits}),
        "contract": "ReactionIntent_Contract_v1.md L321-343 + "
                    "feature_contract.forbidden_model_inputs (yaml)",
        **_preset_fields(preset_obj, explicit),
    }
    if hits and raise_on_violation:
        raise IllegalFeatureError([k for k, _ in hits])
    return report


# ---------------------------------------------------------------------------
# Positive input allowlist (v2 ORA-2 item 3)
# ---------------------------------------------------------------------------

#: Frozen positive set of forward-input keys.  Sources:
#: ``config/campaigns/pes2ts_30h_20261020.yaml`` ``feature_contract`` —
#: ``f2_content: [pos, elements, K, degK, 无序端点上下文, B_src/B_tgt/ΔB_intent]``,
#: ``f3_extra: [current-distance RBF(12)]``, reaction-intent fields
#: (``G_src/G_tgt``, CV, intent_role, signed intent — ReactionIntent Contract
#: v1 S2/S6), plus charge/multiplicity and graph_current/union bookkeeping.
#: Groups are kept separate purely for evidence; membership is exact-key.
FORWARD_INPUT_ALLOWLIST_GROUPS: dict[str, tuple[str, ...]] = {
    "current_geometry": (
        "pos", "xyz", "coords", "r_ij", "dist_cur", "rbf", "rbf12",
        "current_distance_rbf", "angle", "dihedral", "cv_current",
    ),
    "atomic_numbers": (
        "z", "elements", "atomic_numbers", "charge", "mult", "formal_charge",
        "spin_mult",
    ),
    "graph_current_union": (
        "edge_index", "edge_attr", "adjacency", "graph_index",
        "graph_current", "graph_union", "bond_order", "degK", "deg_k",
        "is_key", "key_index", "k",
    ),
    "reaction_intent_fields": (
        "b_src", "b_tgt", "delta_b_intent", "abs_delta", "b_abs",
        "is_changed", "intent_role", "signed_intent", "direction_encoding",
        "cv_target_delta", "cv_spec", "endpoint_names", "g_src", "g_tgt",
    ),
}

#: frozen union (exact key membership, not substring matching).
FORWARD_INPUT_ALLOWLIST: frozenset[str] = frozenset(
    key for group in FORWARD_INPUT_ALLOWLIST_GROUPS.values() for key in group
)

FORWARD_INPUT_ALLOWLIST_SOURCE = (
    "config/campaigns/pes2ts_30h_20261020.yaml feature_contract "
    "(f2_content + f3_extra + reaction_intent fields), commit 29f0e25"
)


def check_input_allowlist(
    batch: Mapping[str, Any],
    allowed_keys: frozenset[str] | set[str] | None = None,
    *,
    preset: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Whitelist gate: forward inputs must be a subset of the frozen set.

    Blacklist (:func:`check_input_legality`) and whitelist are BOTH enforced;
    the T1.4 gate-report entry carries both results.
    """
    preset_obj, explicit = resolve_preset(preset, strict=strict)
    allowed = FORWARD_INPUT_ALLOWLIST if allowed_keys is None else set(allowed_keys)
    keys = {str(k) for k in batch}
    outside = sorted(keys - set(allowed))
    return {
        "check": "input_allowlist",
        "enforcement": "allowlist",
        "passed": not outside,
        "violations": outside,
        "present_allowed": sorted(keys & set(allowed)),
        "allowed_size": len(allowed),
        "source": FORWARD_INPUT_ALLOWLIST_SOURCE,
        **_preset_fields(preset_obj, explicit),
    }


# ---------------------------------------------------------------------------
# Intent-off full mask (PD-6, v2 ORA-2 item 4)
# ---------------------------------------------------------------------------

#: The five classes PD-6 requires intent-off to mask (yaml
#: ``feature_contract.intent_off_mask``).  Values are lower-cased key
#: substrings covering the canonical field names and their common aliases.
INTENT_OFF_MASK_CLASSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ordered_source_target", (
        "b_src", "b_tgt", "src_graph", "tgt_graph", "source_graph",
        "target_graph", "ordered_src", "ordered_tgt", "endpoint_order",
    )),
    ("signed_graph_edits", (
        "delta_b_intent", "db_intent", "signed_edit", "edit_sign",
        "delta_b", "bond_delta",
    )),
    ("intent_role", ("intent_role", "role_intent", "edge_role")),
    ("directed_cv_descriptors", (
        "cv_target_delta", "cv_delta", "cv_direction", "directed_cv",
        "xi_delta", "cv_tgt", "cv_src",
    )),
    ("implicit_direction_encodings", (
        "signed_intent", "direction_encoding", "intent_sign", "dir_encoding",
        "q_sign", "intent_dir",
    )),
)

#: Legal unordered chemical topology that intent-off must PRESERVE
#: (yaml ``feature_contract.preserved_off_context``: 合法无向化学拓扑).
PRESERVED_UNORDERED_CONTEXT_KEYS: tuple[str, ...] = (
    "pos", "z", "elements", "edge_index", "graph_index", "adjacency",
    "b_abs", "abs_delta", "is_changed", "degK", "deg_k",
)


def _is_masked(value: Any) -> bool:
    """A value counts as masked when it is absent/empty/zero (PD-6: 置零)."""
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return True
        return bool((value == 0).all().item())
    if isinstance(value, np.ndarray):
        return value.size == 0 or bool(np.all(value == 0))
    if isinstance(value, (list, tuple)):
        return len(value) == 0 or all(_is_masked(v) for v in value)
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, str):
        return value == ""
    if isinstance(value, Mapping):
        return len(value) == 0 or all(_is_masked(v) for v in value.values())
    return False


def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return list(a) == list(b)
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return False
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - fall back to identity comparison
        return a is b


def check_intent_off_mask(
    intent_on: Mapping[str, Any],
    intent_off: Mapping[str, Any] | Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    *,
    preset: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """PD-6: the intent-off variant must mask ALL five classes and PRESERVE
    the unordered chemical topology.

    ``intent_off`` may be the variant batch or a callable generator (e.g. the
    dummy model's :func:`make_intent_off_batch`).  Missing variant -> state
    ``blocked`` (never a silent pass).
    """
    preset_obj, explicit = resolve_preset(preset, strict=strict)
    fields = _preset_fields(preset_obj, explicit)

    if intent_off is None:
        return {
            "check": "intent_off_mask",
            "passed": False,
            "state": "blocked",
            "block_reason": "intent_off_variant_missing",
            "classes": {},
            "preserved_ok": None,
            **fields,
        }
    if callable(intent_off):
        intent_off = intent_off(intent_on)
    off: Mapping[str, Any] = intent_off

    on_keys = {str(k): k for k in intent_on}
    classes: dict[str, Any] = {}
    unmasked: list[str] = []
    missing_classes: list[str] = []
    for cls, patterns in INTENT_OFF_MASK_CLASSES:
        matched = [
            name for name in on_keys
            if any(pat in name.lower() for pat in patterns)
        ]
        still: list[str] = []
        for name in matched:
            key = on_keys[name]
            if key in off and not _is_masked(off[key]):
                still.append(name)
        present = bool(matched)
        masked_ok = present and not still
        if not present:
            missing_classes.append(cls)
        if still:
            unmasked.append(cls)
        classes[cls] = {
            "keys": sorted(matched),
            "present_in_on": present,
            "masked": masked_ok if present else None,
            "unmasked_keys": sorted(still),
        }

    preserved_bad: list[str] = []
    preserved_missing: list[str] = []
    preserved_checked: list[str] = []
    for name in PRESERVED_UNORDERED_CONTEXT_KEYS:
        if name not in intent_on:
            continue
        preserved_checked.append(name)
        if name not in off:
            preserved_missing.append(name)
        elif not _same_value(intent_on[name], off[name]):
            preserved_bad.append(name)
    preserved_ok = not preserved_bad and not preserved_missing

    blocked = not classes  # pathological empty contract
    if blocked:
        state = "blocked"
    elif unmasked or not preserved_ok:
        state = "fail"
    else:
        state = "pass"

    return {
        "check": "intent_off_mask",
        "passed": state == "pass",
        "state": state,
        "classes": classes,
        "classes_unmasked": sorted(unmasked),
        "classes_absent_from_on_batch": sorted(missing_classes),
        "class_coverage": f"{len(INTENT_OFF_MASK_CLASSES) - len(missing_classes)}"
                          f"/{len(INTENT_OFF_MASK_CLASSES)}",
        "preserved_checked": sorted(preserved_checked),
        "preserved_violations": sorted(preserved_bad),
        "preserved_missing_in_off": sorted(preserved_missing),
        "preserved_ok": preserved_ok,
        "contract": "PD-6 intent_off_mask + preserved_off_context "
                    "(feature_contract, yaml commit 29f0e25)",
        "on_keys": sorted(intent_on),
        "off_keys": sorted(off),
        **fields,
    }


# ---------------------------------------------------------------------------
# Shared report/preset fields
# ---------------------------------------------------------------------------

def _preset_fields(preset: Preset, explicit: bool) -> dict[str, Any]:
    """Fields every v2 report carries (no silent preset anywhere)."""
    return {
        "harness_version": HARNESS_VERSION,
        "preset": preset.name,
        "preset_explicit": bool(explicit),
        "record_only": bool(preset.record_only),
        "gate_admissible": _admissible(preset, explicit),
        "banner": preset.banner if preset.record_only else None,
        "tolerance": {"rel": preset.rel, "abs": preset.abs},
        "relabel_max_abs": preset.relabel_max_abs,
        "preset_source": preset.source,
    }


def _overall_passed(checks_passed: bool, preset: Preset, explicit: bool) -> bool:
    """v2: numeric pass AND explicit preset AND admissible preset."""
    return bool(checks_passed and _admissible(preset, explicit))


# ---------------------------------------------------------------------------
# Batch helpers (non-mutating)
# ---------------------------------------------------------------------------

def _copy_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.clone() if isinstance(v, torch.Tensor) else v
    return out


def _float3_tensors(batch: Mapping[str, Any]) -> list[str]:
    return [
        k for k, v in batch.items()
        if isinstance(v, torch.Tensor) and v.is_floating_point()
        and v.ndim >= 2 and v.shape[-1] == 3
    ]


#: Keys holding *absolute* coordinates (translated but not differenced).
ABS_POS_KEYS: tuple[str, ...] = ("pos", "pos0", "xyz", "coords")


def _as_output_dict(out: Any) -> dict[str, torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return {"u": out}
    if isinstance(out, Mapping):
        return {k: v for k, v in out.items() if isinstance(v, torch.Tensor)}
    raise TypeError(
        f"model output must be a tensor or a mapping of tensors, got {type(out)}"
    )


def _call_model(model: Callable[..., Any], batch: Mapping[str, Any]):
    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            return model(batch)
    finally:
        if was_training and hasattr(model, "train"):
            model.train()


def _u_vecs(out: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if "u" not in out:
        raise KeyError("model output dict must contain equivariant 'u' [K,3]")
    return out["u"]


def _stats(errors: torch.Tensor | np.ndarray) -> dict[str, float]:
    arr = np.asarray(
        errors.detach().cpu().numpy() if isinstance(errors, torch.Tensor)
        else errors, dtype=np.float64
    ).ravel()
    if arr.size == 0:
        return {"median": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "median": float(np.median(arr)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _threshold(preset: Preset, scale: float) -> float:
    return preset.abs + preset.rel * float(scale)


def _scale(u: torch.Tensor) -> float:
    return float(u.detach().abs().max().item()) if u.numel() else 0.0


def _report(
    check: str,
    errors: torch.Tensor,
    passed: bool,
    preset: Preset,
    explicit: bool,
    scale: float,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "check": check,
        "passed": bool(passed),
        **_preset_fields(preset, explicit),
        "scale": scale,
        "threshold": _threshold(preset, scale),
        **_stats(errors),
        **extra,
    }


# ---------------------------------------------------------------------------
# T1.1-T1.3 geometric/semantic checks
# ---------------------------------------------------------------------------

def check_translation_invariance(
    model: Callable[..., Any],
    batch: Mapping[str, Any],
    *,
    preset: str | None = None,
    strict: bool = True,
    shift: torch.Tensor | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """max|u(x+t) - u(x)|; absolute-coordinate keys are translated."""
    preset_obj, explicit = resolve_preset(preset, strict=strict)
    base = _copy_batch(batch)
    moved = _copy_batch(batch)
    if shift is None:
        gen = torch.Generator().manual_seed(seed)
        shift = (torch.rand(3, generator=gen) - 0.5) * 4.0  # ~[-2,2] A
    for key in ABS_POS_KEYS:
        if key in moved and isinstance(moved[key], torch.Tensor):
            moved[key] = moved[key] + shift.to(moved[key].dtype)

    out_a = _as_output_dict(_call_model(model, base))
    out_b = _as_output_dict(_call_model(model, moved))
    u_a, u_b = _u_vecs(out_a), _u_vecs(out_b)
    errors = (u_a - u_b).detach().abs()
    scale = max(_scale(u_a), _scale(u_b))
    return _report(
        "translation_invariance", errors,
        bool(errors.max().item() <= _threshold(preset_obj, scale)),
        preset_obj, explicit, scale,
        max_abs_delta=float(errors.max().item()),
        shift=shift.tolist(),
    )


def _random_rotation(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=gen, dtype=torch.float64))
    q = q @ torch.diag(torch.sign(torch.diag(r)))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.float()


def check_rotation_equivariance(
    model: Callable[..., Any],
    batch: Mapping[str, Any],
    *,
    preset: str | None = None,
    strict: bool = True,
    rotation: torch.Tensor | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-atom |R u(x) - u(R x)| for a random proper rotation R."""
    preset_obj, explicit = resolve_preset(preset, strict=strict)
    R = rotation if rotation is not None else _random_rotation(seed)
    base = _copy_batch(batch)
    rot = _copy_batch(batch)
    for key in _float3_tensors(rot):
        rot[key] = rot[key] @ R.T.to(rot[key].dtype)

    out_a = _as_output_dict(_call_model(model, base))
    out_b = _as_output_dict(_call_model(model, rot))
    u_a, u_b = _u_vecs(out_a), _u_vecs(out_b)
    Rl = R.to(u_a.dtype)
    per_atom = ((u_a @ Rl.T) - u_b).detach().norm(dim=-1)  # [K]
    scale = max(_scale(u_a), _scale(u_b))
    return _report(
        "rotation_equivariance", per_atom,
        bool(per_atom.numel() and per_atom.max().item() <= _threshold(preset_obj, scale)),
        preset_obj, explicit, scale,
        per_atom_errors=per_atom.tolist(),
        rotation=R.tolist(),
    )


def check_permutation_equivariance(
    model: Callable[..., Any],
    batch: Mapping[str, Any],
    *,
    preset: str | None = None,
    strict: bool = True,
    permutation: torch.Tensor | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """|u(P x) un-permuted - u(x)| for a within-graph atom permutation P."""
    preset_obj, explicit = resolve_preset(preset, strict=strict)
    base = _copy_batch(batch)
    pos = base["pos"]
    n = int(pos.shape[0])

    if permutation is None:
        gen = torch.Generator().manual_seed(seed)
        graph = base.get("graph_index")
        perm = torch.empty(n, dtype=torch.long)
        if graph is None:
            perm = torch.randperm(n, generator=gen)
        else:
            graph = graph.long()
            for gid in torch.unique(graph):
                idx = torch.nonzero(graph == gid, as_tuple=False).ravel()
                perm[idx] = idx[torch.randperm(idx.numel(), generator=gen)]
    else:
        perm = permutation.long()
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n, dtype=torch.long)

    permuted = _copy_batch(batch)
    for key, v in list(permuted.items()):
        if key == "key_index" or key == "edge_index":
            continue  # index tensors: remapped below, not gathered
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == n:
            permuted[key] = v[perm]
    if "edge_index" in permuted and isinstance(permuted["edge_index"], torch.Tensor):
        permuted["edge_index"] = inv[permuted["edge_index"].long()]
    if "key_index" in permuted and isinstance(permuted["key_index"], torch.Tensor):
        # values re-point at the atoms' post-permutation rows; row ORDER of
        # the key list (hence of model outputs) is preserved
        permuted["key_index"] = inv[permuted["key_index"].long()]

    out_a = _as_output_dict(_call_model(model, base))
    out_b = _as_output_dict(_call_model(model, permuted))
    u_a = _u_vecs(out_a)
    u_b = _u_vecs(out_b)
    # Row alignment: with `key_index` the key-list ORDER is fixed (values
    # re-pointed), so outputs align directly; without it outputs follow
    # storage order and must be un-permuted (u_back[perm] = u_b).
    if "key_index" in batch:
        errors = (u_a - u_b).detach().abs()
    else:
        u_back = torch.empty_like(u_a)
        u_back[perm] = u_b
        errors = (u_a - u_back).detach().abs()
    scale = max(_scale(u_a), _scale(u_b))
    return _report(
        "permutation_equivariance", errors,
        bool(errors.numel() and errors.max().item() <= _threshold(preset_obj, scale)),
        preset_obj, explicit, scale,
        max_abs_delta=float(errors.max().item()),
        permutation=perm.tolist(),
    )


def check_relabel_invariance(
    model: Callable[..., Any],
    batch: Mapping[str, Any],
    *,
    preset: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Endpoint display-name swap: outputs must be bit-identical.

    30h plan S3.3/S7.4 T1.1: swapping the two endpoint display names must
    leave the outputs **bit-identical**; hard threshold ``max|du| <= 1e-6``
    (feature_contract.tolerances.relabel_invariance_max_abs, PD-2).
    """
    preset_obj, explicit = resolve_preset(preset, strict=strict)
    names = batch.get("endpoint_names")
    if names is None:
        return {
            "check": "relabel_invariance",
            "passed": False,
            "state": "blocked",
            "block_reason": "endpoint_names_missing",
            **_preset_fields(preset_obj, explicit),
        }
    base = _copy_batch(batch)
    swapped = _copy_batch(batch)
    swapped["endpoint_names"] = tuple(reversed(tuple(names)))

    out_a = _as_output_dict(_call_model(model, base))
    out_b = _as_output_dict(_call_model(model, swapped))
    u_a, u_b = _u_vecs(out_a), _u_vecs(out_b)
    errors = (u_a - u_b).detach().abs()
    max_delta = float(errors.max().item()) if errors.numel() else 0.0
    bit_identical = all(
        torch.equal(out_a[k], out_b[k]) for k in out_a
        if k in out_b and out_a[k].shape == out_b[k].shape
    ) and set(out_a) == set(out_b)
    tol = preset_obj.relabel_max_abs
    return {
        "check": "relabel_invariance",
        "passed": bool(bit_identical and max_delta <= tol),
        **_preset_fields(preset_obj, explicit),
        "relabel_abs_tol": tol,
        "bit_identical": bool(bit_identical),
        "max_abs_delta": max_delta,
        "scale": max(_scale(u_a), _scale(u_b)),
        **_stats(errors),
    }


def check_reverse_sensitivity(
    model: Callable[..., Any],
    batch: Mapping[str, Any],
    *,
    preset: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Signed-intent flip (T1.2): outputs must CHANGE, no exact-negative req."""
    preset_obj, explicit = resolve_preset(preset, strict=strict)
    if "signed_intent" not in batch:
        return {
            "check": "reverse_sensitivity",
            "passed": False,
            "state": "blocked",
            "block_reason": "signed_intent_missing",
            **_preset_fields(preset_obj, explicit),
        }
    base = _copy_batch(batch)
    flipped = _copy_batch(batch)
    flipped["signed_intent"] = -base["signed_intent"]

    out_a = _as_output_dict(_call_model(model, base))
    out_b = _as_output_dict(_call_model(model, flipped))
    u_a, u_b = _u_vecs(out_a), _u_vecs(out_b)
    errors = (u_a - u_b).detach().abs()
    scale = max(_scale(u_a), _scale(u_b))
    # "changed" must exceed numerical noise under the active preset;
    # exact negative (u_flip == -u) is NOT required (T1.2).
    min_change = max(preset_obj.abs, 1e-12)
    changed = bool(errors.numel() and errors.max().item() > min_change)
    exact_negative = bool(
        errors.numel()
        and torch.allclose(u_b, -u_a, atol=preset_obj.abs, rtol=0.0)
    )
    return _report(
        "reverse_sensitivity", errors, changed,
        preset_obj, explicit, scale,
        outputs_changed=changed,
        exact_negative_observed=exact_negative,
        exact_negative_required=False,
    )


# ---------------------------------------------------------------------------
# NaN -> typed invalid (GeometryPolicyContract v1 S4.3)
# ---------------------------------------------------------------------------

def check_nan_typed_invalid(
    model: Callable[..., Any],
    batch: Mapping[str, Any],
    *,
    preset: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """NaN coordinates must yield a typed invalid, never a fabricated action."""
    preset_obj, explicit = resolve_preset(preset, strict=strict)
    nan_batch = _copy_batch(batch)
    key = "pos" if "pos" in nan_batch else _float3_tensors(nan_batch)[0]
    nan_batch[key] = torch.full_like(nan_batch[key], float("nan"))

    outcome = "fabricated_action"
    code: str | None = None
    try:
        out = _call_model(model, nan_batch)
    except Exception as exc:  # noqa: BLE001 - classified below
        typed = isinstance(exc, TypedInvalidError) or (
            "invalid" in type(exc).__name__.lower()
            and getattr(exc, "code", None)
        )
        outcome = "typed_invalid_raised" if typed else "untyped_exception"
        code = getattr(exc, "code", None) or type(exc).__name__
    else:
        if isinstance(out, Mapping) and (
            out.get("invalid") is True or out.get("status") == "invalid"
        ):
            outcome = "typed_invalid_returned"
            code = str(out.get("reason") or out.get("code") or "INVALID")
        else:
            tensors = _as_output_dict(out)
            nonfinite = any(
                not torch.isfinite(v).all().item()
                for v in tensors.values() if v.is_floating_point()
            )
            outcome = "non_finite_output" if nonfinite else "fabricated_action"

    return {
        "check": "nan_typed_invalid",
        "passed": outcome.startswith("typed_invalid"),
        "outcome": outcome,
        "code": code,
        **_preset_fields(preset_obj, explicit),
        "contract": "GeometryPolicyContract_v1.md S4.3 (L121/L125/L170/L183)",
    }


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------

def run_feature_gates(
    model: Callable[..., Any],
    batch: Mapping[str, Any],
    *,
    preset: str | None = None,
    strict: bool = True,
    seed: int = 0,
    intent_off: Mapping[str, Any]
    | Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run every gate the batch supports; aggregate pass/fail (v2).

    ``passed`` is fail-closed: numeric pass AND ``preset_explicit`` AND
    ``gate_admissible``.  A record-only or non-explicit preset can therefore
    never produce ``passed=true``.
    """
    resolved, explicit = resolve_preset(preset, strict=strict)
    checks: dict[str, Any] = {
        "input_legality": check_input_legality(
            batch, preset=preset, strict=strict),
        "input_allowlist": check_input_allowlist(
            batch, preset=preset, strict=strict),
        "nan_typed_invalid": check_nan_typed_invalid(
            model, batch, preset=preset, strict=strict),
        "translation_invariance": check_translation_invariance(
            model, batch, preset=preset, strict=strict, seed=seed),
        "rotation_equivariance": check_rotation_equivariance(
            model, batch, preset=preset, strict=strict, seed=seed),
        "permutation_equivariance": check_permutation_equivariance(
            model, batch, preset=preset, strict=strict, seed=seed),
    }
    if "endpoint_names" in batch:
        checks["relabel_invariance"] = check_relabel_invariance(
            model, batch, preset=preset, strict=strict)
    if "signed_intent" in batch:
        checks["reverse_sensitivity"] = check_reverse_sensitivity(
            model, batch, preset=preset, strict=strict)
    if intent_off is not None:
        checks["intent_off_mask"] = check_intent_off_mask(
            batch, intent_off, preset=preset, strict=strict)

    numeric = all(_is_pass(r) for r in checks.values())
    return {
        "harness_version": HARNESS_VERSION,
        "preset": resolved.name,
        "preset_explicit": explicit,
        "preset_source": resolved.source,
        "record_only": resolved.record_only,
        "gate_admissible": _admissible(resolved, explicit),
        "banner": resolved.banner if resolved.record_only else None,
        "numeric_pass": numeric,
        "passed": _overall_passed(numeric, resolved, explicit),
        "checks": checks,
    }


def _is_pass(report: Mapping[str, Any]) -> bool:
    if "state" in report and report.get("state") not in (None, "pass"):
        return False
    return bool(report.get("passed"))


# ---------------------------------------------------------------------------
# G1 gate report (v2 ORA-2 item 5)
# ---------------------------------------------------------------------------

#: test id -> check keys (30h plan S7.4 table; T1.4b/T1.4c are WP-F2 extras).
GATE_REPORT_TESTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("T1.1", ("relabel_invariance",)),
    ("T1.2", ("reverse_sensitivity",)),
    ("T1.3", ("translation_invariance", "rotation_equivariance",
              "permutation_equivariance")),
    ("T1.4", ("input_legality", "nan_typed_invalid")),
    ("T1.4b", ("input_allowlist",)),
    ("T1.4c", ("intent_off_mask",)),
)

_VALID_STATES = ("pass", "fail", "blocked", "not_run")


def _entry_state(reports: Sequence[Mapping[str, Any]]) -> str:
    states = []
    for r in reports:
        st = r.get("state")
        if st not in _VALID_STATES:
            st = "pass" if _is_pass(r) else "fail"
        states.append(st)
    if "fail" in states:
        return "fail"
    if "blocked" in states:
        return "blocked"
    if "not_run" in states:
        return "not_run"
    return "pass"


def build_gate_report(
    model: Callable[..., Any],
    batch: Mapping[str, Any],
    *,
    preset: str | None = None,
    campaign_yaml: str | Path | None = DEFAULT_CAMPAIGN_YAML,
    strict: bool = True,
    seed: int = 0,
    intent_off: Mapping[str, Any]
    | Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the G1 gate report: T1.1-T1.4 + T1.4b + T1.4c.

    Every entry carries ``state`` (``pass|fail|blocked|not_run``), its full
    check evidence, the preset and the campaign-yaml sha256 that bound the
    tolerances.  The G1 consumer can diff ``yaml_sha256`` later.
    """
    run = run_feature_gates(
        model, batch, preset=preset, strict=strict, seed=seed,
        intent_off=intent_off)

    campaign_block: dict[str, Any] | None = None
    yaml_ok = False
    yaml_sha: str | None = None
    block_reason: str | None = None
    if campaign_yaml is not None:
        try:
            contract, yaml_sha = load_campaign_feature_contract(campaign_yaml)
            resolved, _ = resolve_preset(preset, strict=strict)
            tol_block = verify_preset_against_yaml(resolved, contract, yaml_sha)
            yaml_ok = True
            campaign_block = {
                "path": str(campaign_yaml),
                "sha256": yaml_sha,
                "tolerances": tol_block,
                "feature_contract_present": True,
                "parse_mode": contract.get("_parse_mode", "whole_file"),
            }
        except PresetContractError as exc:
            block_reason = str(exc).split(":", 1)[0]
            campaign_block = {
                "path": str(campaign_yaml),
                "sha256": yaml_sha,
                "feature_contract_present": False,
                "error": str(exc),
            }
    else:
        block_reason = "CAMPAIGN_YAML_NOT_SUPPLIED"

    checks = run["checks"]
    entries: list[dict[str, Any]] = []
    for test_id, names in GATE_REPORT_TESTS:
        evidence = {n: checks[n] for n in names if n in checks}
        missing = [n for n in names if n not in checks]
        if missing:
            # a required check is unavailable (prerequisite batch field or
            # intent-off variant absent) -> blocked, per the gate-state
            # vocabulary of the 30h plan S7.1/S7.3 (fail-closed: the test
            # must never be silently skipped or passed)
            state = "blocked"
        else:
            state = _entry_state(list(evidence.values()))
        entries.append({
            "test_id": test_id,
            "state": state,
            "checks": list(evidence),
            "missing_checks": missing,
            "evidence": evidence,
            "preset": run["preset"],
            "yaml_sha256": yaml_sha,
            "gate_admissible": run["gate_admissible"],
        })

    entry_states = [e["state"] for e in entries]
    if "fail" in entry_states:
        overall = "fail"
    elif not yaml_ok or "blocked" in entry_states or not run["gate_admissible"]:
        overall = "blocked"
    elif "not_run" in entry_states:
        overall = "not_run"
    else:
        overall = "pass"

    banner = run["banner"]
    if overall != "pass" and banner is None:
        banner = block_reason
    return {
        "schema": "pes2ts_feature_gate_report_v1",
        "harness_version": HARNESS_VERSION,
        "gate": "G1-Feature",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_repr": type(model).__name__,
        "batch_keys": sorted(str(k) for k in batch),
        "seed": seed,
        "preset": {
            "name": run["preset"],
            "explicit": run["preset_explicit"],
            "record_only": run["record_only"],
            "gate_admissible": run["gate_admissible"],
            "source": run["preset_source"],
        },
        "campaign_yaml": campaign_block,
        "yaml_sha256": yaml_sha,
        "banner": banner,
        "tests": entries,
        "state": overall,
        "numeric_pass": run["numeric_pass"],
        "gate_admissible": run["gate_admissible"],
        "passed": overall == "pass",
    }


# ---------------------------------------------------------------------------
# CLI (G1 binding: --gate-report / --preset / --campaign-yaml)
# ---------------------------------------------------------------------------

def _main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    import sys

    from .dummy_equivariant_model import (
        make_dummy_batch,
        make_dummy_model,
        make_intent_off_batch,
    )

    parser = argparse.ArgumentParser(
        prog="equivariance_harness",
        description="Run the feature-gate harness on the bundled dummy model "
                    "(harness v2: normative preset binding, fail-closed).",
    )
    parser.add_argument(
        "--preset", default=None,
        choices=sorted(PRESETS),
        help="tolerance preset. NORMATIVE: '30h_v1.1_normative' (rel 1e-5 + "
             "abs 1e-6, relabel 1e-6; bound to the campaign yaml feature_"
             "contract). 'r3_v3_record' is record-only (abs 1e-4): reports "
             "carry gate_admissible=false + banner, never a pass gate. "
             "Required (strict default).",
    )
    parser.add_argument(
        "--campaign-yaml", default=None,
        help=f"campaign yaml with feature_contract.tolerances "
             f"(default when --gate-report is used: {DEFAULT_CAMPAIGN_YAML})",
    )
    parser.add_argument(
        "--gate-report", default=None, metavar="OUT",
        help="write the G1 gate-report JSON (T1.1-T1.4, T1.4b, T1.4c) here",
    )
    parser.add_argument(
        "--intent-off-mode", default="full",
        choices=("full", "partial", "nuke", "off"),
        help="how the dummy intent-off variant is generated (default: full; "
             "'off' leaves T1.4c blocked)",
    )
    parser.add_argument(
        "--legacy-non-strict", action="store_true",
        help="opt in to the v1 warn-and-fallback preset behaviour "
             "(diagnostics only; reports get gate_admissible=false)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--print-full", action="store_true",
        help="print the full report JSON to stdout even with --gate-report",
    )
    args = parser.parse_args(argv)
    strict = not args.legacy_non_strict

    try:
        resolve_preset(args.preset, strict=strict)
    except (PresetRequiredError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    model = make_dummy_model(seed=args.seed)
    batch = make_dummy_batch(seed=args.seed)
    campaign_yaml = args.campaign_yaml
    if campaign_yaml is None and args.gate_report:
        campaign_yaml = DEFAULT_CAMPAIGN_YAML
    intent_off = None
    if args.intent_off_mode != "off":
        intent_off = make_intent_off_batch(batch, mode=args.intent_off_mode)

    if args.gate_report:
        report = build_gate_report(
            model, batch, preset=args.preset, campaign_yaml=campaign_yaml,
            strict=strict, seed=args.seed, intent_off=intent_off)
        out = Path(args.gate_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True),
                       encoding="utf-8")
        if args.print_full:
            print(json.dumps(report, indent=2, sort_keys=True))
        print(
            f"gate G1-Feature state={report['state']} "
            f"admissible={report['gate_admissible']} "
            f"preset={report['preset']['name']} "
            f"yaml_sha256={report['yaml_sha256']} -> {out}"
        )
        return 0 if report["passed"] else 1

    report = run_feature_gates(
        model, batch, preset=args.preset, strict=strict, seed=args.seed,
        intent_off=intent_off)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
