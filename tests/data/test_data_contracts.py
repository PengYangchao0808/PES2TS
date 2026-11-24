"""Tests for the 3-layer data-batch contract v2 (``pes2ts_data.contracts``)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pes2ts_data.contracts import (
    BASE_FIELDS,
    CODES,
    COMPAT_FIELDS,
    CONTRACT_REQUIRED_FIELDS,
    CONTRACT_VERSION,
    FIELD_SPECS,
    GRAPH_FIELDS,
    LEGACY_CONTRACT_VERSION,
    LEGACY_FIELD_MAP,
    SPLIT_TAGS,
    BaseBatch,
    ContractViolation,
    FieldSpec,
    GraphBatch,
    Provenance,
    SequenceBatch,
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

def graph_batch(**overrides) -> GraphBatch:
    values = {
        "sample_id": "sample-1",
        "provenance": Provenance("curve-1:train"),
        "pos": np.zeros((3, 3), dtype=np.float32),
        "z": np.array([6, 6, 6], dtype=np.int64),
        "edge_index": np.array([[1, 2], [0, 1]], dtype=np.int64),
    }
    values.update(overrides)
    return GraphBatch(**values)


def graph_mapping(**overrides) -> dict:
    batch = graph_batch()
    mapping = batch.as_mapping()
    mapping.update(overrides)
    return mapping


def sequence_batch(**overrides) -> SequenceBatch:
    n_batch, n_frames, n_feat, q_dim = 2, 4, 8, 6
    values = {
        "sample_id": "traj-batch",
        "provenance": "curve-1:train",
        "frame_feats": np.zeros((n_batch, n_frames, n_feat), dtype=np.float32),
        "static_feats": np.zeros((n_batch, 5), dtype=np.float32),
        "delta_s": np.zeros((n_batch, n_frames), dtype=np.float32),
        "lengths": np.array([n_frames, n_frames], dtype=np.int64),
        "move_target": np.zeros((n_batch, n_frames, q_dim), dtype=np.float32),
        "process_target": np.zeros((n_batch, n_frames), dtype=np.int64),
        "progress_target": np.zeros((n_batch, n_frames), dtype=np.float32),
        "trust_target": np.zeros((n_batch, n_frames), dtype=np.float32),
    }
    values.update(overrides)
    return SequenceBatch(**values)


def legacy_trajectory_batch() -> dict:
    batch = sequence_batch().as_mapping()
    batch.update({
        "process_label": batch.pop("process_target"),
        "process_progress": batch.pop("progress_target"),
        "trust_label": batch.pop("trust_target"),
        "provenance_id": batch.pop("provenance"),
        "frame_mask": np.ones((2, 4), dtype=bool),
        "move_mask": np.ones((2, 4), dtype=bool),
    })
    return batch


# ---------------------------------------------------------------------------
# v2 metadata
# ---------------------------------------------------------------------------

def test_contract_v2_metadata():
    assert CONTRACT_VERSION == "2.0.0"
    assert LEGACY_CONTRACT_VERSION == "1.0.0"
    assert set(BASE_FIELDS) == {"sample_id", "targets", "split_tag", "provenance"}
    assert set(GRAPH_FIELDS) == {
        "pos", "z", "edge_index", "endpoint_mask", "sym_pair_idx",
        "target_xyz", "sample_weight"}
    assert set(COMPAT_FIELDS) == {
        "frame_feats", "static_feats", "lengths", "delta_s", "move_target",
        "process_target", "progress_target", "trust_target"}
    assert set(FIELD_SPECS) == set(BASE_FIELDS) | set(GRAPH_FIELDS) | set(COMPAT_FIELDS)
    assert CONTRACT_REQUIRED_FIELDS == ("pos", "z", "edge_index")
    assert "train" in SPLIT_TAGS and "confirm" in SPLIT_TAGS
    assert {"MISSING_FIELD", "TRUTH_FIELD_FORBIDDEN", "PROVENANCE_MISSING"} <= set(CODES)


def test_field_spec_layers_and_flags():
    assert FIELD_SPECS["sample_id"].layer == "base"
    assert FIELD_SPECS["pos"].layer == "graph"
    assert FIELD_SPECS["frame_feats"].layer == "sequence"
    assert FIELD_SPECS["provenance"].mutable is False
    assert FIELD_SPECS["target_xyz"].provenance_sensitive is True
    assert FIELD_SPECS["move_target"].provenance_sensitive is True
    assert FIELD_SPECS["sample_weight"].provenance_sensitive is True
    with pytest.raises(ValueError):
        FieldSpec(name="x", dtype="complex", ndim=1, required=False,
                  description="bad")


def test_legacy_field_map_is_one_to_one():
    assert set(LEGACY_FIELD_MAP) == set(COMPAT_FIELDS)
    assert LEGACY_FIELD_MAP["process_target"] == "process_label"
    assert LEGACY_FIELD_MAP["progress_target"] == "process_progress"
    assert LEGACY_FIELD_MAP["trust_target"] == "trust_label"


# ---------------------------------------------------------------------------
# Three batch layers
# ---------------------------------------------------------------------------

def test_base_batch_validates_identity_layer_only():
    batch = BaseBatch(sample_id="s1", targets={"move": np.zeros(3)},
                      split_tag="train", provenance=Provenance("curve-1:train"))
    report = validate_batch(batch)
    assert report["layer"] == "base"
    assert report["sample_id"] == "s1"
    assert report["targets"] == ["move"]
    assert report["provenance"] == "curve-1:train"
    assert report["required"] == ["sample_id", "provenance"]


def test_graph_batch_validates_with_core_and_identity():
    batch = graph_batch(target_xyz=np.zeros((3, 3), dtype=np.float32),
                        sample_weight=np.ones(3, dtype=np.float32))
    report = validate_batch(batch)
    assert report["layer"] == "graph"
    assert report["fields"]["pos"]["shape"] == (3, 3)
    assert set(report["required"]) == set(CONTRACT_REQUIRED_FIELDS) | {
        "sample_id", "provenance"}
    assert report["provenance"] == "curve-1:train"


def test_sequence_batch_validates_compat_layer():
    report = validate_batch(sequence_batch(), require_core=False)
    assert report["layer"] == "sequence"
    assert set(report["compat_fields"]) == set(COMPAT_FIELDS)
    assert report["fields"]["lengths"]["dtype"].startswith("int")


def test_plain_mappings_infer_the_layer():
    assert validate_batch(graph_mapping())["layer"] == "graph"
    sequence_mapping = sequence_batch().as_mapping()
    assert validate_batch(sequence_mapping, require_core=False)["layer"] == "sequence"
    base_mapping = {"sample_id": "s1", "provenance": "tok"}
    assert validate_batch(base_mapping, require_core=False)["layer"] == "base"


def test_standard_batch_is_deprecated_but_still_validates():
    with pytest.warns(DeprecationWarning):
        batch = StandardBatch(**graph_mapping())
    report = validate_batch(batch)
    assert report["layer"] == "legacy"
    assert report["deprecated"] is True


def test_torch_tensors_are_accepted_by_duck_typing():
    torch = pytest.importorskip("torch")
    batch = graph_batch(
        pos=torch.zeros(3, 3, dtype=torch.float32),
        z=torch.tensor([6, 6, 6], dtype=torch.int64),
        edge_index=torch.tensor([[1, 2], [0, 1]], dtype=torch.int64))
    assert validate_batch(batch)["fields"]["z"]["shape"] == (3,)


# ---------------------------------------------------------------------------
# Typed rejections
# ---------------------------------------------------------------------------

def test_identity_and_core_requirements_raise_missing_field():
    with pytest.raises(ContractViolation) as excinfo:
        validate_batch({})
    assert excinfo.value.code == "MISSING_FIELD"

    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(
            {"sample_id": "s1", "provenance": "tok",
             "pos": np.zeros((3, 3), dtype=np.float32)},
            require_core=True)
    assert excinfo.value.code == "MISSING_FIELD"
    assert "z" in excinfo.value.detail

    with pytest.raises(ContractViolation) as excinfo:
        validate_batch({"pos": np.zeros((3, 3), np.float32),
                        "z": np.zeros(3, np.int64),
                        "edge_index": np.zeros((2, 2), np.int64)},
                       require_identity=True)
    assert excinfo.value.code == "MISSING_FIELD"
    assert "sample_id" in excinfo.value.detail

    # Pure legacy compat batches can opt out of the identity requirement.
    validate_batch(sequence_batch().as_mapping(), require_core=False,
                   require_identity=False)


def test_model_required_field_is_enforced():
    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(graph_batch(), required=("target_xyz",))
    assert excinfo.value.code == "MISSING_FIELD"
    assert "target_xyz" in excinfo.value.detail


@pytest.mark.parametrize("overrides,code", [
    ({"pos": np.zeros((3, 3), dtype=np.int64)}, "BAD_DTYPE"),
    ({"z": np.ones(3, dtype=np.float32)}, "BAD_DTYPE"),
    ({"pos": np.zeros((3, 2), dtype=np.float32)}, "BAD_SHAPE"),
    ({"edge_index": np.zeros((3, 3), dtype=np.int64)}, "BAD_SHAPE"),
    ({"z": np.array([6, 6], dtype=np.int64)}, "SHAPE_MISMATCH"),
    ({"edge_index": np.array([[0, 5], [1, 2]], dtype=np.int64)}, "BAD_INDEX"),
])
def test_graph_field_violations_are_typed(overrides, code):
    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(graph_batch(**overrides))
    assert excinfo.value.code == code


def test_base_field_violations_are_typed():
    with pytest.raises(ContractViolation) as excinfo:
        validate_batch({"sample_id": "s1", "provenance": "tok", "targets": [1, 2]})
    assert excinfo.value.code == "TARGETS_INVALID"

    with pytest.raises(ContractViolation) as excinfo:
        validate_batch({"sample_id": "s1", "provenance": "tok",
                        "split_tag": "everything"})
    assert excinfo.value.code == "SPLIT_TAG_INVALID"

    with pytest.raises(ContractViolation) as excinfo:
        validate_batch({"sample_id": "s1", "provenance": ""})
    assert excinfo.value.code == "PROVENANCE_INVALID"

    with pytest.raises(ContractViolation) as excinfo:
        validate_batch({"sample_id": "s1",
                        "provenance": Provenance("tok", sha256="abc")})
    assert excinfo.value.code == "PROVENANCE_INVALID"

    with pytest.raises(ContractViolation) as excinfo:
        validate_batch({"sample_id": "", "provenance": "tok"})
    assert excinfo.value.code == "BAD_VALUE"


def test_provenance_missing_for_sensitive_fields():
    mapping = graph_mapping(target_xyz=np.zeros((3, 3), dtype=np.float32))
    mapping.pop("provenance")
    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(mapping, require_identity=False)
    assert excinfo.value.code == "PROVENANCE_MISSING"
    # Explicit opt-out for callers whose loader audited provenance upstream.
    validate_batch(mapping, require_identity=False, require_provenance=False)


def test_truth_fields_are_forbidden_r1():
    with pytest.raises(ContractViolation) as excinfo:
        validate_batch({"ts_geometry_xyz": np.zeros((3, 3))})
    assert excinfo.value.code == "TRUTH_FIELD_FORBIDDEN"

    with pytest.raises(ContractViolation) as excinfo:
        validate_batch({"ground_truth_energy": np.zeros(3)})
    assert excinfo.value.code == "TRUTH_FIELD_FORBIDDEN"

    assert is_truth_field("ts_geometry")
    assert is_truth_field("GROUND-TRUTH-energy")
    assert not is_truth_field("trust_target")


def test_assert_fields_and_truth_requirements():
    assert_fields(graph_batch(), ("pos", "z"))
    with pytest.raises(ContractViolation) as excinfo:
        assert_fields(graph_batch(), ("target_xyz",))
    assert excinfo.value.code == "MISSING_FIELD"
    with pytest.raises(ContractViolation) as excinfo:
        assert_fields(graph_batch(), ("ts_geometry",))
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
        validate_batch(graph_batch(pos=object()))
    assert excinfo.value.code == "BAD_DTYPE"

    with pytest.raises(ContractViolation) as excinfo:
        validate_batch({1: np.zeros(3)})
    assert excinfo.value.code == "BAD_BATCH_TYPE"


def test_unknown_fields_tolerated_by_default_but_rejectable():
    batch = graph_mapping(unknown_extra=np.zeros(3))
    validate_batch(batch)
    with pytest.raises(ContractViolation) as excinfo:
        validate_batch(batch, allow_unknown=False)
    assert excinfo.value.code == "UNKNOWN_FIELD"


# ---------------------------------------------------------------------------
# Deprecated compat shim
# ---------------------------------------------------------------------------

def test_to_standard_shim_maps_legacy_keys_and_warns():
    legacy = legacy_trajectory_batch()
    with pytest.warns(DeprecationWarning):
        batch = to_standard(legacy)
    assert isinstance(batch, SequenceBatch)
    assert batch.process_target is legacy["process_label"]
    assert batch.progress_target is legacy["process_progress"]
    assert batch.trust_target is legacy["trust_label"]
    assert batch.provenance is legacy["provenance_id"]
    assert batch.sample_id == legacy["sample_id"]
    # Contract-external legacy keys (masks) are not part of the batch contract.
    assert "frame_mask" not in batch.as_mapping()


def test_from_standard_shim_round_trips_legacy_keys_and_warns():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        batch = to_standard(legacy_trajectory_batch())
    with pytest.warns(DeprecationWarning):
        legacy = from_standard(batch)
    assert legacy["process_label"] is batch.process_target
    assert legacy["process_progress"] is batch.progress_target
    assert legacy["trust_label"] is batch.trust_target
    assert legacy["provenance"] == batch.provenance


def test_compat_shims_reject_truth_and_non_mapping():
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ContractViolation) as excinfo:
            to_standard({"reference_ts_xyz": np.zeros((3, 3))})
    assert excinfo.value.code == "TRUTH_FIELD_FORBIDDEN"

    with pytest.warns(DeprecationWarning):
        with pytest.raises(ContractViolation) as excinfo:
            to_standard(["not", "a", "mapping"])
    assert excinfo.value.code == "BAD_BATCH_TYPE"

    with pytest.warns(DeprecationWarning):
        with pytest.raises(ContractViolation) as excinfo:
            from_standard(object())
    assert excinfo.value.code == "BAD_BATCH_TYPE"


def test_base_batch_as_mapping_skips_none():
    batch = BaseBatch(sample_id="s1", provenance="tok")
    mapping = batch.as_mapping()
    assert mapping == {"sample_id": "s1", "provenance": "tok"}
    assert batch.field_names() == ("sample_id", "targets", "split_tag", "provenance")
    assert graph_batch().field_names()[:4] == batch.field_names()
