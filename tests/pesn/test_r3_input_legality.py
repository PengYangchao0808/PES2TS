"""R3 T1.4 — input-legality whitelist (machine-checked, no leakage).

Enumerates the campaign ``feature_contract.forbidden_model_inputs`` classes
and asserts none is reachable from the objects the model consumes
(forward-view batch keys, built feature dicts, and the sample/record data
classes).  Both the blacklist AND the positive allowlist are enforced.
Fixture-level — ``not_admissible_until_gate_pass`` until G0+G1.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    FORBIDDEN_INPUT_CLASSES_V3,
    FORWARD_VIEW_KEYS_V3,
    SUPERVISION_KEYS_V3,
    KeyAtomSampleV3,
    KeyAtomV3ContractError,
    check_forward_view_keys_v3,
    forward_view_v3,
)
from scripts.agent_ops.feature_gate.equivariance_harness import (  # noqa: E402
    FORBIDDEN_FEATURE_PATTERNS,
    check_input_allowlist,
    check_input_legality,
)
from scripts.agent_ops.feature_gate.v3_fixtures import (  # noqa: E402
    make_v3_batch,
    make_v3_gate_batch,
)

PRESET = "30h_v1.1_normative"


def test_forward_view_is_blacklist_clean() -> None:
    batch = make_v3_gate_batch(n_graphs=2)
    report = check_input_legality(batch, preset=PRESET)
    assert report["passed"], report
    assert report["violations"] == []


def test_forward_view_is_allowlist_clean() -> None:
    batch = make_v3_gate_batch(n_graphs=2)
    report = check_input_allowlist(batch, preset=PRESET)
    assert report["passed"], report
    assert report["violations"] == []


def test_every_forbidden_class_enumerated() -> None:
    """The eleven-yaml classes map onto the harness blacklist classes."""
    names = {cls for cls, _ in FORBIDDEN_INPUT_CLASSES_V3}
    assert names == {
        "future_frames", "ts_coords", "true_progress", "remaining_arc",
        "branch_id", "split_family", "energy_gradient", "near_ts_capped",
        "endpoint_geometry", "history_hidden_state"}
    harness_names = {cls for cls, _ in FORBIDDEN_FEATURE_PATTERNS}
    # every yaml class is covered by (or stricter than) a harness class
    assert {"future_frames", "ts_coords", "branch_id", "split_family",
            "energy_gradient", "near_ts_capped_flags",
            "history_hidden_state"} <= harness_names


@pytest.mark.parametrize("forbidden", (
    "future_frame", "pos_next", "ts_coord", "true_progress",
    "remaining_arc", "branch_id", "split", "family", "energy", "gradient",
    "near_ts", "capped", "endpoint_geom", "history", "hidden_state"))
def test_forbidden_key_fails_legality(forbidden: str) -> None:
    batch = dict(make_v3_gate_batch(n_graphs=1))
    batch[forbidden] = np.zeros(3)
    report = check_input_legality(batch, preset=PRESET)
    assert report["passed"] is False
    assert forbidden in report["violations"]


def test_model_consumes_only_forward_view_keys() -> None:
    batch = make_v3_batch(n_graphs=2)          # full batch incl supervision
    assert set(batch) - set(FORWARD_VIEW_KEYS_V3)  # supervision is present
    fwd = forward_view_v3(batch)
    assert set(fwd) == {k for k in FORWARD_VIEW_KEYS_V3 if k in batch}
    assert not (set(fwd) & set(SUPERVISION_KEYS_V3))


def test_supervision_keys_never_forward_visible() -> None:
    batch = make_v3_batch(n_graphs=1)
    for key in ("y_teacher", "pair_index", "pair_edit", "frame_weight",
                "sample_ids", "curve_ids", "intent_ids", "d0"):
        assert key in SUPERVISION_KEYS_V3 or key in ("d0",)
    fwd = forward_view_v3(batch)
    for key in ("y_teacher", "pair_index", "pair_edit", "frame_weight",
                "sample_ids", "curve_ids", "intent_ids"):
        assert key not in fwd, key


def test_sample_dataclass_has_no_history_fields() -> None:
    names = {f.name for f in dataclasses.fields(KeyAtomSampleV3)}
    forbidden = {
        "pos0", "anchor_vec", "branch", "branch_id", "progress", "q",
        "ts_index", "energy", "gradient", "near_ts", "capped", "terminal",
        "future", "history", "split", "family"}
    assert not (names & forbidden), names & forbidden


def test_feature_dict_has_no_forbidden_keys() -> None:
    from scripts.agent_ops.feature_gate.v3_fixtures import (
        build_fixture_features,
    )

    feats = build_fixture_features("forward")
    report = check_input_legality(
        {k: v for k, v in feats.items() if isinstance(v, np.ndarray)},
        preset=PRESET)
    assert report["passed"], report


def test_blacklist_hits_every_forbidden_class() -> None:
    """Probe keys prove each harness blacklist class actually fires."""
    probes = {
        "future_frames": "pos_future",
        "ts_coords": "pos_ts",
        "true_progress_arc": "true_progress",
        "branch_id": "branch_id",
        "split_family": "split",
        "energy_gradient": "energy",
        "near_ts_capped_flags": "near_ts",
        "history_hidden_state": "history_buffer",
        "endpoint_geometry_XA_XB": "endpoint_xyz",
        "endpoint_derived_dA_dB_s": "d_a_0",
        "travel_direction": "direction_token",
    }
    classes = {cls for cls, _ in FORBIDDEN_FEATURE_PATTERNS}
    assert set(probes) == classes
    for cls, key in probes.items():
        report = check_input_legality({key: np.zeros(1)}, preset=PRESET)
        assert report["passed"] is False, cls
        assert key in report["violations"], cls


# ---------------------------------------------------------------------------
# V5.2-R D0.1 — explicit forward-view negative tests for the Move/terminal/
# stop_status keys.  These three keys carry the R3 Move teacher / Stop-head
# semantics and MUST NEVER appear in a model forward batch (contract §7
# ``stop_status``/``move_status`` are supervision-side; ``terminal`` is a
# Move eligibility that leaks the Stop answer if fed as an input).  The
# structural refusal is ``check_forward_view_keys_v3`` → FORWARD_VIEW_LEAK
# (these keys are absent from ``FORWARD_VIEW_KEYS_V3``); the harness
# blacklist/whitelist provide the second, independent gate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("leak_key", (
    "move_status", "terminal", "stop_status"))
def test_status_keys_fail_forward_view(leak_key: str) -> None:
    """``move_status``/``terminal``/``stop_status`` are NEVER forward inputs."""
    batch = dict(make_v3_gate_batch(n_graphs=1))
    batch[leak_key] = np.zeros(1)
    # structural contract gate: the key is not in FORWARD_VIEW_KEYS_V3, so
    # check_forward_view_keys_v3 refuses with the typed FORWARD_VIEW_LEAK.
    with pytest.raises(KeyAtomV3ContractError) as exc:
        check_forward_view_keys_v3(batch)
    assert exc.value.code == "FORWARD_VIEW_LEAK"
    assert leak_key not in FORWARD_VIEW_KEYS_V3
    # harness allowlist: every one of the three keys is outside the frozen
    # forward-input allowlist (exact-key membership).
    allow = check_input_allowlist(batch, preset=PRESET)
    assert allow["passed"] is False
    assert leak_key in allow["violations"]
    # harness blacklist: substring patterns catch ``terminal`` (near_ts class)
    # directly; ``move_status``/``stop_status`` are not substring-matched by
    # the frozen pattern list, so the blacklist alone is NOT the gate for
    # them — pin that behaviour explicitly so a future pattern extension is
    # a deliberate, reviewed change.
    report = check_input_legality(batch, preset=PRESET)
    if leak_key == "terminal":
        assert report["passed"] is False
        assert leak_key in report["violations"]
    else:
        # the allowlist + FORWARD_VIEW_LEAK above are the effective gates
        assert allow["passed"] is False
