"""Tests for the single data-batch contract (``pes2ts_data.contracts``)."""

from __future__ import annotations

import numpy as np
import pytest

from pes2ts_data.contracts import (
    COMPAT_FIELDS,
    CONTRACT_REQUIRED_FIELDS,
    CONTRACT_VERSION,
    FIELD_SPECS,
    LEGACY_FIELD_MAP,
    ContractViolation,
    FieldSpec,
    StandardBatch,
    assert_fields,
    from_standard,
    is_truth_field,
    required_fields,
    to_standard,
    validate_batch,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def minimal_batch(**overrides):
    """A minimal valid geometric batch: pos (3,3), z (3,), edge_index (2,2)."""
    batch = {
        "pos": np.zeros((3, 3), dtype=np.float32),
        "z": np.array([6, 6, 6], dtype=np.int64),
        "edge_index": np.array([[1, 2], [0, 1]], dtype=np.int64),
    }
    batch.update(overrides)
    return batch


def legacy_trajectory_batch():
    """A legacy V5.6 trajectory collate batch (padded [B,T,...] tensors)."""
    n_batch, n_frames, n_feat, q_dim = 2, 4, 8, 6
    return {
        "frame_feats": np.zeros((n_batch, n_frames, n_feat), dtype=np.float32),
        "static_feats": np.zeros((n_batch, 5), dtype=np.float32),
        "delta_s": np.zeros((n_batch, n_frames), dtype=np.float32),
        "lengths": np.array([n_frames, n_frames], dtype=np.int64),
        "frame_mask": np.ones((n_batch, n_frames), dtype=bool),
        "move_target": np.zeros((n_batch, n_frames, q_dim), dtype=np.float32),
        "move_mask": np.ones((n_batch, n_frames), dtype=bool),
        "process_label": np.zeros((n_batch, n_frames), dtype=np.int64),
        "process_mask": np.ones((n_batch, n_frames), dtype=bool),
        "process_progress": np.zeros((n_batch, n_frames), dtype=np.float32),
        "progress_mask": np.ones((n_batch, n_frames), dtype=bool),
        "trust_label": np.zeros((n_batch, n_frames), dtype=np.float32),
        "trust_mask": np.ones((n_batch, n_frames), dtype=bool),
        "q": np.zeros((n_batch, n_frames, q_dim), dtype=np.float32),
        "s": np.zeros((n_batch, n_frames), dtype=np.float32),
        "provenance_id": ["curve-1:train", "curve-2:train"],
    }


# ---------------------------------------------------------------------------
# Field specs
# ---------------------------------------------------------------------------

def test_field_specs_declare_canonical_and_compat_fields():
    canonical = {"pos", "z", "edge_index", "endpoint_mask", "sym_pair_idx",
                 "target_xyz", "sample_weight", "provenance_id"}
    assert canonical <= set(FIELD_SPECS)
    assert set(COMPAT_FIELDS) == {
        "frame_feats", "static_feats", "lengths", "delta_s", "move_target",
        "process_target", "progress_target", "trust_target"}
    assert set(COMPAT_FIELDS) <= set(FIELD_SPECS)
    assert CONTRACT_VERSION


def test_contract_required_fields_are_the_geometric_identity():
    assert CONTRACT_REQUIRED_FIELDS == ("pos", "z", "edge_index")
    assert all(FIELD_SPECS[name].required for name in CONTRACT_REQUIRED_FIELDS)


def test_compat_field_map_is_one_to_one():
    assert set(LEGACY_FIELD_MAP) == set(COMPAT_FIELDS)
    assert len(set(LEGACY_FIELD_MAP.values())) == len(LEGACY_FIELD_MAP)
    assert LEGACY_FIELD_MAP["process_target"] == "process_label"
    assert LEGACY_FIELD_MAP["progress_target"] == "process_progress"
    assert LEGACY_FIELD_MAP["trust_target"] == "trust_label"


def test_provenance_sensitive_fields_are_marked():
    for name in ("target_xyz", "sample_weight", "move_target",
                 "process_target", "progress_target", "trust_target"):
        assert FIELD_SPECS[name].provenance_sensitive is True, name
    assert FIELD_SPECS["provenance_id"].mutable is False


def test_field_spec_rejects_unknown_dtype():
    with pytest.raises(ValueError):
        FieldSpec(name="x", dtype="complex", ndim=1, required=False,
                  description="bad")


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_validate_standard_batch_dataclass():
    batch = StandardBatch(**minimal_batch())
    report = validate_batch(batch)
    assert set(report["fields"]) == {"pos", "z", "edge_index"}
    assert report["provenance_id"] is False
    assert report["required"] == list(CONTRACT_REQUIRED_FIELDS)


def test_validate_plain_mapping_and_torch_tensor():
    report = validate_batch(minimal_batch())
    assert report["fields"]["pos"]["shape"] == (3, 3)

    torch = pytest.importorskip("torch")
    batch = minimal_batch(
        pos=torch.zeros(3, 3, dtype=torch.float32),
        z=torch.tensor([6, 6, 6], dtype=torch.int64),
        edge_index=torch.tensor([[1, 2], [0, 1]], dtype=torch.int64))
    assert validate_batch(batch)["fields"]["z"]["shape"] == (3,)


def test_validate_with_provenance_passes():
    report = validate_batch(minimal_batch(
        target_xyz=np.zeros((3, 3), dtype=np.float32),
        sample_weight=np.ones(3, dtype=np.float32),
        provenance_id="curve-1:train:sha256"))
    assert report["provenance_id"] is True
    assert report["fields"]["target_xyz"]["provenance_sensitive"] is True


# ---------------------------------------------------------------------------
# Typed rejections
# ---------------------------------------------------------------------------

def test_missing_contract_fields_raise_missing_field():
    with pytest.raises(ContractViolation) as excinfo:
        validate_batch({})
    assert excinfo.value.code == "MISSING_FIELD"


def test_model_required_field_is_enforced():
    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(minimal_batch(), required=("target_xyz",))
    assert excinfo.value.code == "MISSING_FIELD"
    assert "target_xyz" in excinfo.value.detail


def test_bad_dtype_raises_coded_error():
    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(minimal_batch(pos=np.zeros((3, 3), dtype=np.int64)))
    assert excinfo.value.code == "BAD_DTYPE"
    assert excinfo.value.field == "pos"

    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(minimal_batch(z=np.ones(3, dtype=np.float32)))
    assert excinfo.value.code == "BAD_DTYPE"


def test_bad_shape_raises_coded_error():
    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(minimal_batch(pos=np.zeros((3, 2), dtype=np.float32)))
    assert excinfo.value.code == "BAD_SHAPE"

    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(minimal_batch(
            edge_index=np.zeros((3, 3), dtype=np.int64)))
    assert excinfo.value.code == "BAD_SHAPE"


def test_shape_mismatch_and_bad_index_raise_coded_errors():
    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(minimal_batch(z=np.array([6, 6], dtype=np.int64)))
    assert excinfo.value.code == "SHAPE_MISMATCH"

    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(minimal_batch(
            edge_index=np.array([[0, 5], [1, 2]], dtype=np.int64)))
    assert excinfo.value.code == "BAD_INDEX"


def test_provenance_missing_raises():
    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(minimal_batch(target_xyz=np.zeros((3, 3), dtype=np.float32)))
    assert excinfo.value.code == "PROVENANCE_MISSING"
    # ... but an explicit opt-out is allowed for loaders that audited provenance.
    validate_batch(minimal_batch(target_xyz=np.zeros((3, 3), dtype=np.float32)),
                   require_provenance=False)


def test_truth_fields_are_forbidden_r1():
    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(minimal_batch(ts_geometry_xyz=np.zeros((3, 3))))
    assert excinfo.value.code == "TRUTH_FIELD_FORBIDDEN"

    with pytest.raises(ContractViolation) as excinfo:
        validate_batch({"ground_truth_energy": np.zeros(3)})
    assert excinfo.value.code == "TRUTH_FIELD_FORBIDDEN"

    assert is_truth_field("ts_geometry")
    assert is_truth_field("GROUND-TRUTH-energy")
    assert not is_truth_field("trust_target")


def test_assert_fields_rejects_missing_and_truth_requirements():
    assert_fields(minimal_batch(), ("pos", "z"))
    with pytest.raises(ContractViolation) as excinfo:
        assert_fields(minimal_batch(), ("target_xyz",))
    assert excinfo.value.code == "MISSING_FIELD"
    with pytest.raises(ContractViolation) as excinfo:
        assert_fields(minimal_batch(), ("ts_geometry",))
    assert excinfo.value.code == "TRUTH_FIELD_FORBIDDEN"


def test_required_fields_helper_preserves_order():
    assert required_fields(("a", "b", "c"), ("b", "c")) == ["a"]
    assert required_fields(("c", "a", "b"), ()) == ["c", "a", "b"]
    assert required_fields(("a", "a"), ("a",)) == []


def test_validation_never_leaks_native_exceptions():
    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(object())
    assert excinfo.value.code == "BAD_BATCH_TYPE"

    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(minimal_batch(pos=object()))
    assert excinfo.value.code == "BAD_DTYPE"

    with pytest.raises(ContractViolation) as excinfo:
        validate_batch({1: np.zeros(3)})
    assert excinfo.value.code == "BAD_BATCH_TYPE"


def test_unknown_fields_tolerated_by_default_but_rejectable():
    batch = minimal_batch(unknown_extra=np.zeros(3))
    validate_batch(batch)
    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(batch, allow_unknown=False)
    assert excinfo.value.code == "UNKNOWN_FIELD"


# ---------------------------------------------------------------------------
# Legacy V5.6 compat bridge
# ---------------------------------------------------------------------------

def test_to_standard_renames_legacy_keys_and_preserves_passthrough():
    legacy = legacy_trajectory_batch()
    snapshot = {key: value for key, value in legacy.items()}
    standard = to_standard(legacy)

    assert standard["process_target"] is legacy["process_label"]
    assert standard["progress_target"] is legacy["process_progress"]
    assert standard["trust_target"] is legacy["trust_label"]
    assert "lengths" in standard
    # Passthrough legacy keys are preserved, never silently dropped (R5).
    for passthrough in ("frame_mask", "move_mask", "process_mask", "q", "s"):
        assert passthrough in standard
    # Input never mutated.
    assert set(legacy) == set(snapshot)


def test_compat_batch_validates_with_require_core_false():
    standard = to_standard(legacy_trajectory_batch())
    report = validate_batch(
        standard,
        required=("frame_feats", "static_feats", "lengths"),
        require_core=False,
    )
    assert set(report["compat_fields"]) == set(COMPAT_FIELDS)
    assert report["provenance_id"] is True


def test_compat_batch_without_provenance_is_rejected():
    legacy = legacy_trajectory_batch()
    legacy.pop("provenance_id")
    standard = to_standard(legacy)
    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(standard, require_core=False)
    assert excinfo.value.code == "PROVENANCE_MISSING"


def test_from_standard_round_trips_compat_keys():
    standard = to_standard(legacy_trajectory_batch())
    legacy = from_standard(standard)
    assert legacy["process_label"] is standard["process_target"]
    assert legacy["process_progress"] is standard["progress_target"]
    assert legacy["trust_label"] is standard["trust_target"]
    assert "lengths" in legacy


def test_from_standard_accepts_standard_batch_dataclass():
    standard = to_standard(legacy_trajectory_batch())
    declared = set(StandardBatch.__dataclass_fields__)
    batch = StandardBatch(**{key: value for key, value in standard.items()
                             if key in declared})
    out = from_standard(batch)
    assert "process_label" in out
    assert "trust_label" in out
    assert "lengths" in out


def test_compat_adapters_reject_truth_and_non_mapping():
    with pytest.raises(ContractViolation) as excinfo:
        to_standard({"reference_ts_xyz": np.zeros((3, 3))})
    assert excinfo.value.code == "TRUTH_FIELD_FORBIDDEN"

    with pytest.raises(ContractViolation) as excinfo:
        to_standard(["not", "a", "mapping"])
    assert excinfo.value.code == "BAD_BATCH_TYPE"

    with pytest.raises(ContractViolation) as excinfo:
        from_standard(object())
    assert excinfo.value.code == "BAD_BATCH_TYPE"


def test_standard_batch_as_mapping_skips_none():
    batch = StandardBatch(**minimal_batch())
    mapping = batch.as_mapping()
    assert set(mapping) == {"pos", "z", "edge_index"}
    assert batch.field_names()[0] == "pos"
