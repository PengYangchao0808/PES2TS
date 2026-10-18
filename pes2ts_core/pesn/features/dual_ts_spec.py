"""Frozen dual-TS feature spec v3 (WP3-A + v4.1 WP1 observation-only inputs).

This module is the SINGLE pre-registration surface for every frozen decision
of the ``pesn_dual_ts_features_v3`` feature spec (G4 entry condition; v4.1
WP1 removed the teacher-label-derived node-scalar channels, 7 → 5).  It
must stay import-clean (numpy + the contract module only) and must be the
place where feature-consuming code reads its constants — never inline
literals.

Frozen decisions recorded here (G3 oracle checklist items 7a-7f + disputed-5
rule (2)):

* **s_g rule** — ONE shared positive scalar over the fit partition:
  ``s_g = median(||g_i||_2)`` over every atom of every fit state
  (Hartree/Å).  Applied as ``g̃ = g / s_g``: direction and relative
  magnitude preserved, the SAME scale on all 3 axes (plan §I.7 — never
  per-axis scales, never unit-direction-only).
* **ΔE rule** — ``ΔE_t = E(state) − E(episode.start_state_id)`` read from
  the SAME bound observation table (no recomputation); the episode start is
  the endpoint state of that branch.  Frozen scale
  ``s_ΔE = 95th percentile of |ΔE|`` over the fit population (numpy
  ``percentile`` linear interpolation); normalized ``ΔẼ = ΔE / s_ΔE``.
* **Disputed-5 rule** — for the reactions whose task carries
  ``eligibility.bond_authority == "disputed"`` (RXN_0000000160/0756/0778/
  1076/1174), the encoder-input endpoint graph is the deterministic
  covalent-radius geometry perception (SAME radii and tolerance as the
  ``dual_ts_preprocess`` fallback: Cordero 2008 single-bond radii,
  tolerance 0.45 Å) run on that branch's episode-start (endpoint frame)
  exported coordinates; authority bonds are kept for bookkeeping only.
  All other reactions consume the authority endpoint graphs.  The rule is
  applied symmetrically to A/B; a per-sample ``bond_source`` flag
  (``authority`` | ``covalent_fallback``) keeps the two populations
  separable — the authority-vs-fallback sensitivity slice is WP4 metrics
  scope and the two are never silently mixed.
* **Distance basis** — v1 ``gaussian_distance_basis`` parameters reused
  verbatim: cutoff 5.0 Å, 32 Gaussian centers on ``linspace(0, cutoff,
  n_basis)``, shared scale ``cutoff / n_basis``; distances in Å.
* **Loss eligibility** — near-zero-action atoms (per-atom action norm
  ``||a_i||_2 < 0.01`` Å) are masked OUT of direction losses; the terminal
  ``teacher_endpoint`` state of each branch carries no action and is
  excluded from action losses entirely.
* **Input layout** — exact ``declared_layout`` with per-block dims; the
  contract enforces ``sum(declared_layout) == input_dim`` and the spec
  content digest binds the whole record.
* **Support domain / missing masks** — every WP2 fit50 state has a bound
  ``status="ok"`` observation (6067/6067, zero nulls), so no missing-mask
  channel is populated in this wave; the mask channels remain declared for
  future waves (missing = ``null + reason``, never zeros).

Scale VALUES are estimated only by
:func:`estimate_fit_scales_from_records` from the frozen fit population at
freeze time (the descriptive G3 medians are NOT the frozen values); the
digested instance is materialized by :func:`build_features_spec` and saved
by the build script.  Everything else on this module is a rule, frozen
before any feature code consumes it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.dual_ts import (
    DualTSFeaturesSpec,
    FEATURES_SCHEMA_VERSION,
    validate_record,
)

#: Version of THIS freeze (recorded in support_domain for traceability).
DUAL_TS_FEATURES_FREEZE_VERSION: Final[str] = "pesn_dual_ts_features_v3_wp41_freeze_v0"

# ---------------------------------------------------------------------------
# Frozen rules (pre-registered before any feature consumption; G3 (7a)-(7f))
# ---------------------------------------------------------------------------

#: G3 (7a): ONE shared positive scalar over all axes; estimation population
#: = the fit wave, ALL states included (TS terminals and inserted midpoints
#: declared in-population); estimator = median over per-atom gradient norms.
S_G_RULE: Final[str] = (
    "median_per_atom_gradient_norm_fit_all_states_v0"
)

#: G3 (7b): ΔE binding (contract-frozen definition) + fit-only scale rule.
DELTA_E_DEFINITION: Final[str] = "E - E_episode_start"
S_DELTA_E_RULE: Final[str] = (
    "p95_abs_delta_e_fit_all_states_numpy_linear_interpolation_v0"
)
S_DELTA_E_PERCENTILE: Final[float] = 95.0

#: G3 (7c): distance expansion — v1 parameters reused verbatim (Å units).
DISTANCE_BASIS_KIND: Final[str] = "gaussian_v1_reused"
DISTANCE_CUTOFF_ANGSTROM: Final[float] = 5.0
DISTANCE_N_BASIS: Final[int] = 32

#: G3 (2) disputed-5: covalent-radius fallback for encoder input (kept
#: identical to the dual_ts_preprocess fallback so the WP2 cross-check and
#: this rule cannot drift apart); authority bonds for bookkeeping only.
DISPUTED_BOND_RULE: Final[str] = (
    "covalent_radius_fallback_on_disputed_authority_v0"
)
BOND_SOURCE_AUTHORITY: Final[str] = "authority"
BOND_SOURCE_COVALENT_FALLBACK: Final[str] = "covalent_fallback"

#: G3 (7e): loss eligibility thresholds/masks.
NEAR_ZERO_ACTION_THRESHOLD_ANGSTROM: Final[float] = 0.01
NEAR_ZERO_MASK_RULE: Final[str] = (
    "per_atom_action_norm_below_threshold_masked_out_of_direction_loss_v0"
)
TEACHER_ENDPOINT_ACTION_RULE: Final[str] = (
    "teacher_endpoint_states_excluded_from_action_loss_v0"
)

#: G3 (7f): consumed gradient precision source = the round-10 teacher
#: observation refs (full precision stays in the receipts).
GRADIENT_PRECISION_SOURCE: Final[str] = (
    "round10_teacher_observation_refs_v0"
)

#: Exchange-symmetric fusion rule (exact string bound into the spec): the
#: SHARED encoder is applied to each endpoint's per-atom/per-edge features
#: and the two embeddings are fused by SUM + ELEMENTWISE-MAX
#: ``fused = concat([h_a + h_b, max(h_a, h_b)])``.  Never per-edge sorting
#: across endpoints (which would lose endpoint association of bond orders).
FUSION_RULE: Final[str] = (
    "shared_encoder_sum_plus_elementwise_max_v0:"
    "fused=concat(h_a+h_b, max(h_a,h_b));"
    "never_per_edge_sorting_across_endpoints"
)

#: Translation removal rule for the equivariant X channel: applied at
#: consume time (Å): ``X_c = X - mean_atoms(X)`` (unweighted centroid).
TRANSLATION_REMOVAL_RULE: Final[str] = (
    "unweighted_centroid_removed_at_consume_time_v0"
)

# ---------------------------------------------------------------------------
# Frozen input layout (per-block dims; contract checks the sum)
# ---------------------------------------------------------------------------


def declared_layout_blocks() -> dict[str, int]:
    """Frozen per-block dims of the state input layout.

    * ``node_scalar`` (per atom, 5): ``[z_id, grad_norm_scaled,
      delta_e_scaled, charge, multiplicity]`` — ``z_id`` is the integer
      embedding id channel (element category embedding is the encoder's
      job).  Observation-only: the former ``is_teacher_endpoint`` /
      ``near_zero_action_mask`` channels were REMOVED in v4.1 WP1 (teacher
      labels survive only as supervision, not network input).
    * ``endpoint_atom_feat`` (4): per-endpoint per-atom topological
      features ``[degree, bond_order_sum, changed_bond_degree,
      stereo_known_degree]`` fed to the SHARED endpoint encoder.
    * ``endpoint_edge_feat`` (4): per-endpoint per-union-edge features
      ``[bond_present, bond_order, changed_flag, stereo_known]`` fed to the
      SHARED endpoint encoder.
    * ``equivariant_vec_channels`` (2): the ``X`` and ``g`` vector channels.
    """
    return {
        "node_scalar": 5,
        "endpoint_atom_feat": 4,
        "endpoint_edge_feat": 4,
        "equivariant_vec_channels": 2,
    }


NODE_SCALAR_NAMES: Final[tuple[str, ...]] = (
    "z_id",
    "grad_norm_scaled",
    "delta_e_scaled",
    "charge",
    "multiplicity",
)
ENDPOINT_ATOM_FEAT_NAMES: Final[tuple[str, ...]] = (
    "degree", "bond_order_sum", "changed_bond_degree", "stereo_known_degree",
)
ENDPOINT_EDGE_FEAT_NAMES: Final[tuple[str, ...]] = (
    "bond_present", "bond_order", "changed_flag", "stereo_known",
)

# ---------------------------------------------------------------------------
# Missing-masks policy (declared for future waves; zero nulls in WP2 fit50)
# ---------------------------------------------------------------------------

MISSING_MASKS_POLICY: Final[Mapping[str, object]] = {
    "observation_missing_policy": "null_plus_reason_never_zero",
    "wp2_fit50_status": "all 6067 states bound status=ok; zero nulls; "
                        "mask channels declared but uniformly present "
                        "this wave (consumers treat all as present)",
    "channels_declared": ["observation_present", "gradient_present",
                          "energy_present"],
}

# ---------------------------------------------------------------------------
# Scale estimation from the frozen fit population
# ---------------------------------------------------------------------------


def estimate_fit_scales_from_records(
    records: Iterable[Mapping[str, object]],
) -> dict[str, float]:
    """Frozen estimators over the fit population (G3 7a/7b).

    ``records`` must iterate ONE entry per fit state, each carrying
    ``gradient`` (N,3) array-like Hartree/Å and ``delta_e`` (float, Hartree)
    — the values already bound to the episode start via the observation
    table.  Returns ``{"s_g": ..., "s_delta_e": ...}``:

    * ``s_g`` = median of per-atom gradient norms ``||g_i||_2`` over ALL
      states/atoms of the population (one scalar, all axes).
    * ``s_delta_e`` = 95th percentile of ``|delta_e|`` (numpy linear
      interpolation).
    """
    norms: list[float] = []
    abs_delta: list[float] = []
    n_states = 0
    for rec in records:
        g = np.asarray(rec["gradient"], dtype=float).reshape(-1, 3)
        if g.size == 0:
            raise ValueError("empty gradient record in fit population")
        if not np.isfinite(g).all():
            raise ValueError("non-finite gradient in fit population")
        norms.extend(np.linalg.norm(g, axis=1).tolist())
        d = float(rec["delta_e"])
        if not np.isfinite(d):
            raise ValueError("non-finite delta_e in fit population")
        abs_delta.append(abs(d))
        n_states += 1
    if n_states == 0:
        raise ValueError("empty fit population")
    s_g = float(np.median(np.asarray(norms, dtype=float)))
    s_delta_e = float(np.percentile(
        np.asarray(abs_delta, dtype=float), S_DELTA_E_PERCENTILE))
    if not (np.isfinite(s_g) and s_g > 0.0):
        raise ValueError(f"s_g estimator produced {s_g!r}; must be finite > 0")
    if not (np.isfinite(s_delta_e) and s_delta_e > 0.0):
        raise ValueError(
            f"s_delta_e estimator produced {s_delta_e!r}; must be finite > 0")
    return {"s_g": s_g, "s_delta_e": s_delta_e}


# ---------------------------------------------------------------------------
# Spec instance construction (contract class + validate_record + digest)
# ---------------------------------------------------------------------------


def build_features_spec(
    *,
    s_g: float,
    s_delta_e: float,
    element_vocab: Sequence[int],
    population: Mapping[str, object],
    n_atoms_range: tuple[int, int],
) -> tuple[DualTSFeaturesSpec, dict]:
    """Materialize the content-digested ``pesn_dual_ts_features_v3`` instance.

    Builds the payload with every frozen rule/value above, validates it via
    the contract registry (:func:`validate_record`), round-trips it through
    ``from_payload`` and returns ``(spec, payload)``.  ``s_g``/``s_delta_e``
    must come from :func:`estimate_fit_scales_from_records` over the frozen
    fit population.
    """
    layout = declared_layout_blocks()
    vocab = sorted(int(z) for z in set(int(v) for v in element_vocab))
    support_domain: dict[str, object] = {
        "freeze_version": DUAL_TS_FEATURES_FREEZE_VERSION,
        "elements": vocab,
        "n_atoms_range": [int(n_atoms_range[0]), int(n_atoms_range[1])],
        "coordinates_unit": "angstrom",
        "gradient_unit": "hartree/angstrom",
        "energy_unit": "hartree",
        "observation_method": "GFN2-xTB",
        "a_max_angstrom": 0.1,
        "gradient_precision_source": GRADIENT_PRECISION_SOURCE,
        "distance_basis": {
            "kind": DISTANCE_BASIS_KIND,
            "cutoff_angstrom": float(DISTANCE_CUTOFF_ANGSTROM),
            "n_basis": int(DISTANCE_N_BASIS),
            "centers": "linspace(0, cutoff_angstrom, n_basis)",
            "scale": "cutoff_angstrom / n_basis",
            "distance_unit": "angstrom",
        },
        "s_g_rule": S_G_RULE,
        "s_delta_e_rule": S_DELTA_E_RULE,
        "delta_energy_binding": (
            "E(state) - E(episode.start_state_id) read from the SAME bound "
            "observation table; episode start = the endpoint state of the "
            "branch; no recomputation"),
        "bond_rule": {
            "disputed_rule": DISPUTED_BOND_RULE,
            "radii_source": "cordero_2008_single_bond (dual_ts_preprocess)",
            "tolerance_angstrom": 0.45,
            "applied_symmetrically": True,
            "bond_source_flag_per_sample": [
                BOND_SOURCE_AUTHORITY, BOND_SOURCE_COVALENT_FALLBACK],
            "sensitivity_slice": "wp4_metrics_reporting_not_silently_mixed",
        },
        "fusion_rule": FUSION_RULE,
        "translation_removal": TRANSLATION_REMOVAL_RULE,
        "near_zero_mask_rule": NEAR_ZERO_MASK_RULE,
        "teacher_endpoint_action_rule": TEACHER_ENDPOINT_ACTION_RULE,
        "declared_layout_doc": {
            "node_scalar": list(NODE_SCALAR_NAMES),
            "endpoint_atom_feat": list(ENDPOINT_ATOM_FEAT_NAMES),
            "endpoint_edge_feat": list(ENDPOINT_EDGE_FEAT_NAMES),
            "equivariant_vec_channels": ["X", "g"],
        },
        "population": dict(population),
    }
    masks: dict[str, object] = {
        "near_zero_action_threshold_angstrom":
            float(NEAR_ZERO_ACTION_THRESHOLD_ANGSTROM),
        "near_zero_masked_from_direction_loss": True,
        "teacher_endpoint_excluded_from_action_loss": True,
        "missing_masks": dict(MISSING_MASKS_POLICY),
    }
    payload: dict[str, object] = {
        "schema_version": FEATURES_SCHEMA_VERSION,
        "encoding": "dual_graph_exchange_symmetric_v0",
        "equivariant_vectors": ["X", "g"],
        "delta_energy_definition": DELTA_E_DEFINITION,
        "frozen_scales": {
            "s_g": float(s_g),
            "s_delta_e": float(s_delta_e),
        },
        "masks": masks,
        "support_domain": support_domain,
        "declared_layout": {k: int(v) for k, v in layout.items()},
        "input_dim": int(sum(layout.values())),
    }
    # Contract validation through the registry (validate_record) AND the
    # typed class round-trip; both must accept the exact payload saved.
    validate_record(FEATURES_SCHEMA_VERSION, payload)
    spec = DualTSFeaturesSpec.from_payload(payload)
    if spec.content_digest() != DualTSFeaturesSpec.from_payload(
            spec.to_payload()).content_digest():
        raise ValueError("features spec payload round-trip changed the digest")
    return spec, payload


# ---------------------------------------------------------------------------
# Spec I/O with digest verification
# ---------------------------------------------------------------------------


def save_spec_payload(payload: Mapping[str, object], path: str | Path) -> str:
    """Save the spec payload (plus its digest) as JSON; returns the digest."""
    from pes2ts_core.pesn.contracts.dual_ts import content_digest_of

    record = {
        "spec": dict(payload),
        "spec_digest": content_digest_of(dict(payload)),
    }
    text = json.dumps(record, indent=1, sort_keys=True,
                      ensure_ascii=False, allow_nan=False)
    Path(path).write_text(text, encoding="utf-8")
    return str(record["spec_digest"])


def load_spec_payload(path: str | Path) -> tuple[DualTSFeaturesSpec, dict]:
    """Load a saved spec, verifying the recorded content digest."""
    from pes2ts_core.pesn.contracts.dual_ts import content_digest_of

    record = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = dict(record["spec"])
    digest = content_digest_of(payload)
    if digest != str(record["spec_digest"]):
        raise ValueError(
            f"features spec digest mismatch in {path}: recorded "
            f"{record['spec_digest']!r} != recomputed {digest!r}")
    spec = DualTSFeaturesSpec.from_payload(payload)
    return spec, payload


__all__ = [
    "BOND_SOURCE_AUTHORITY",
    "BOND_SOURCE_COVALENT_FALLBACK",
    "DELTA_E_DEFINITION",
    "DISPUTED_BOND_RULE",
    "DISTANCE_BASIS_KIND",
    "DISTANCE_CUTOFF_ANGSTROM",
    "DISTANCE_N_BASIS",
    "DUAL_TS_FEATURES_FREEZE_VERSION",
    "ENDPOINT_ATOM_FEAT_NAMES",
    "ENDPOINT_EDGE_FEAT_NAMES",
    "FUSION_RULE",
    "GRADIENT_PRECISION_SOURCE",
    "MISSING_MASKS_POLICY",
    "NEAR_ZERO_ACTION_THRESHOLD_ANGSTROM",
    "NEAR_ZERO_MASK_RULE",
    "NODE_SCALAR_NAMES",
    "S_DELTA_E_PERCENTILE",
    "S_DELTA_E_RULE",
    "S_G_RULE",
    "TEACHER_ENDPOINT_ACTION_RULE",
    "TRANSLATION_REMOVAL_RULE",
    "build_features_spec",
    "declared_layout_blocks",
    "estimate_fit_scales_from_records",
    "load_spec_payload",
    "save_spec_payload",
]
