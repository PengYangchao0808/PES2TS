"""B0/B6 acceptance: typed input qualification, prediction contracts, batch order."""

from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.pesn.contracts.inputs import (
    CoordinateDef,
    CoordinateSpec,
    InputRejected,
    TaskEdit,
    TaskGraph,
    BondTerm,
    build_pesn_input,
    geometry_content_hash,
)
from pes2ts_core.pesn.contracts.predictions import (
    CAPABILITY_KEYS,
    CAP_Q_STOP,
    MUST_BE_DISABLED_CAPS,
    PREDICTION_SCHEMA_VERSION,
    PredictionRejected,
    ScalarEstimate,
    TeacherPrediction,
    UnsupportedCapability,
    VectorEstimate,
    direction_available,
    research_capability_map,
    validate_capability_map,
)
from pes2ts_core.pesn.contracts.artifacts import (
    ArtifactError,
    BUNDLE_FORBIDDEN_KEYS,
    validate_bundle_manifest,
    validate_split_manifest,
)


def _task_graph() -> TaskGraph:
    return TaskGraph(
        schema_version="pesn_task_graph_v1",
        atomic_numbers=(6, 1, 1),
        atom_map=(1, 2, 3),
        reactant_bonds=(BondTerm(0, 1, 1.0),),
        product_bonds=(BondTerm(0, 2, 1.0),),
        edits=(TaskEdit("broken", 0, 1, 1.0, 0.0), TaskEdit("formed", 0, 2, 0.0, 1.0)),
        source="test",
    )


def _coord_spec() -> CoordinateSpec:
    return CoordinateSpec(
        schema_version="pesn_coordinate_spec_v1",
        coords=(
            CoordinateDef("distance", (0, 1), "angstrom", "driver", 1.0, 0.0, "a"),
            CoordinateDef("distance", (0, 2), "angstrom", "driver", 1.0, 0.0, "b"),
        ),
        spec_version="t1",
    )


def _coords() -> np.ndarray:
    return np.asarray([[0.0, 0, 0], [1.1, 0, 0], [-2.0, 0, 0]], dtype=float)


def test_valid_input_builds_and_hashes_stably() -> None:
    first = build_pesn_input(
        input_id="x", task_graph=_task_graph(), coordinate_spec=_coord_spec(), coordinates=_coords()
    )
    second = build_pesn_input(
        input_id="x", task_graph=_task_graph(), coordinate_spec=_coord_spec(), coordinates=_coords()
    )
    assert first.input_hash == second.input_hash
    assert first.geometry_hash == geometry_content_hash(_coords())


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"units": "bohr"}, "UNITS_UNKNOWN"),
        ({"coordinates": [[0, 0, 0], [1, 0, 0]]}, "ATOM_COUNT_MISMATCH"),
        ({"coordinates": [[0, 0, np.nan], [1, 0, 0], [0, 1, 0]]}, "NONFINITE_COORDINATES"),
        ({"input_id": ""}, "INPUT_ID_MISSING"),
    ],
)
def test_input_rejections_are_typed(kwargs: dict, code: str) -> None:
    with pytest.raises(InputRejected) as excinfo:
        build_pesn_input(
            input_id=kwargs.get("input_id", "x"),
            task_graph=_task_graph(),
            coordinate_spec=_coord_spec(),
            coordinates=kwargs.get("coordinates", _coords()),
            units=kwargs.get("units", "angstrom"),
        )
    assert excinfo.value.code == code


def test_task_graph_map_and_element_validation() -> None:
    with pytest.raises(InputRejected) as excinfo:
        TaskGraph(
            schema_version="pesn_task_graph_v1",
            atomic_numbers=(6, 1),
            atom_map=(1, 1),
            reactant_bonds=(),
            product_bonds=(),
            edits=(),
            source="t",
        ).validate()
    assert excinfo.value.code == "MAP_NOT_BIJECTION"

    with pytest.raises(InputRejected) as excinfo:
        TaskGraph(
            schema_version="pesn_task_graph_v1",
            atomic_numbers=(6, 119),
            atom_map=(1, 2),
            reactant_bonds=(),
            product_bonds=(),
            edits=(),
            source="t",
        ).validate()
    assert excinfo.value.code == "ELEMENT_INVALID"


def test_unknown_schema_rejected_everywhere() -> None:
    with pytest.raises(InputRejected):
        build_pesn_input(
            input_id="x",
            task_graph=TaskGraph(
                "pesn_task_graph_v9", (1,), (1,), (), (), (), "t"
            ),
            coordinate_spec=_coord_spec(),
            coordinates=[[0, 0, 0]],
        )


def test_research_capability_map_shape() -> None:
    caps = research_capability_map(
        progress=True, path_proximity=True, teacher_direction=False, coordinate_activity=False
    )
    assert set(caps) == set(CAPABILITY_KEYS)
    for key in MUST_BE_DISABLED_CAPS:
        assert caps[key] == "disabled"
    with pytest.raises(UnsupportedCapability):
        validate_capability_map({**caps, CAP_Q_STOP: "enabled"})


def test_prediction_contract_rejects_malformed_heads() -> None:
    good = TeacherPrediction(
        schema_version=PREDICTION_SCHEMA_VERSION,
        input_hash="a" * 64,
        model_id="m",
        model_hash="b" * 64,
        feature_hash="c" * 64,
        signed_progress=ScalarEstimate(True, 0.1),
        path_proximity=ScalarEstimate(True, 0.1),
        path_score=ScalarEstimate(True, 0.2),
        teacher_direction=VectorEstimate(False, (0.0, 0.0), (False, False), "head_disabled"),
        coordinate_activity=VectorEstimate(False, (0.0, 0.0), (False, False), "head_disabled"),
    )
    good.validate()
    with pytest.raises(PredictionRejected):
        TeacherPrediction(
            **{
                **{
                    "schema_version": good.schema_version,
                    "input_hash": good.input_hash,
                    "model_id": good.model_id,
                    "model_hash": good.model_hash,
                    "feature_hash": good.feature_hash,
                    "signed_progress": ScalarEstimate(True, None),
                    "path_proximity": good.path_proximity,
                    "path_score": good.path_score,
                    "teacher_direction": good.teacher_direction,
                    "coordinate_activity": good.coordinate_activity,
                }
            }
        ).validate()


def test_unavailable_head_reasons_are_closed_set() -> None:
    with pytest.raises(PredictionRejected):
        ScalarEstimate(False, None, "not_a_reason").validate()
    with pytest.raises(PredictionRejected):
        VectorEstimate(True, (1.0,), (True,), "why").validate()


def test_direction_available_requires_nonzero_vector() -> None:
    assert direction_available([0.0, 0.0]) is False
    assert direction_available([0.0, 0.3]) is True
    assert direction_available([]) is False


def _bundle_manifest() -> dict:
    from pes2ts_core.pesn.contracts.predictions import research_capability_map

    return {
        "schema_version": "pesn_model_bundle_v1",
        "bundle_format": "pesn_bundle_format_v1",
        "model_id": "pesn-irc-v0.1.0-rc.1",
        "model_hash": "d" * 64,
        "feature_spec_sha256": "e" * 64,
        "label_spec_sha256": "f" * 64,
        "normalization_sha256": "1" * 64,
        "weights_sha256": "d" * 64,
        "training_run_ids": ["run1"],
        "capabilities": research_capability_map(
            progress=True, path_proximity=True, teacher_direction=True, coordinate_activity=True
        ),
        "model_config": {"hidden": 64, "n_layers": 3, "n_basis": 16},
        "model_card": "model_card.md",
        "checksums": "checksums.json",
        "research_only": True,
    }


def test_bundle_manifest_validation_and_forbidden_keys() -> None:
    manifest = _bundle_manifest()
    validate_bundle_manifest(manifest)
    for key in BUNDLE_FORBIDDEN_KEYS:
        with pytest.raises(ArtifactError):
            validate_bundle_manifest({**manifest, key: "leak"})


def test_split_manifest_requires_frozen_and_no_conflicts() -> None:
    base = {
        "schema_version": "pesn_irc_split_v1",
        "parent_split_sha256": "a" * 64,
        "grouping_version": "pesn_grouping_v1",
        "group_members_sha256": "b" * 64,
        "subsets": {"train_fit": ["RXN_1"]},
        "family_holdout": {"reaction_ids": []},
        "frozen": True,
        "known_group_cross_split_conflicts": 0,
    }
    validate_split_manifest(base)
    with pytest.raises(ArtifactError):
        validate_split_manifest({**base, "known_group_cross_split_conflicts": 3})
    with pytest.raises(ArtifactError):
        validate_split_manifest({**base, "frozen": False})
