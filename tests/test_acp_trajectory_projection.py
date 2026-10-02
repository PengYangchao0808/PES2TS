from __future__ import annotations

from pathlib import Path
import sys

import pytest

from pes2ts_core.contracts import seal_document
from pes2ts_core.demo import synthetic_objects
from pes2ts_core.integration.acp.adapter import acp_result_to_path_bundle
from pes2ts_core.integration.acp.trajectory import (
    ACPFrameContractError,
    path_bundle_to_acp_pes_profile,
    project_path_bundle_to_acp_graph,
)
from pes2ts_core.ranking import rank_path_bundle


def _acp_contract_root(tmp_path: Path) -> Path:
    frames_path = tmp_path / "src" / "acp" / "results" / "frames.py"
    frames_path.parent.mkdir(parents=True)
    (frames_path.parent / "__init__.py").write_text("", encoding="utf-8")
    (frames_path.parent.parent / "__init__.py").write_text("", encoding="utf-8")
    frames_path.write_text(
        """from dataclasses import dataclass, field
from typing import Any
@dataclass(frozen=True)
class TrajectoryFrame:
    frame_id: str; label: str; frame_index: int; x: float | None; energy: float | None
    status: str = 'unknown'; geometry_ref: str = ''; metadata: dict[str, Any] = field(default_factory=dict)
    def to_node(self, node_type: str):
        return {'id': self.frame_id, 'label': self.label, 'type': node_type,
                'frame_index': self.frame_index, 'x': self.x, 'energy': self.energy,
                'status': self.status, 'geometry_ref': self.geometry_ref, 'metadata': dict(self.metadata)}
@dataclass(frozen=True)
class TrajectoryAnnotation:
    id: str; type: str; label: str; frame_index: int; x: float | None; y: float | None
    status: str = ''; geometry_ref: str = ''; selected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    def to_annotation(self):
        return {'id': self.id, 'type': self.type, 'label': self.label,
                'frame_index': self.frame_index, 'x': self.x, 'y': self.y,
                'status': self.status, 'geometry_ref': self.geometry_ref,
                'selected': self.selected, 'metadata': dict(self.metadata)}
@dataclass(frozen=True)
class ViewSpec:
    node_type: str
def view_spec(name: str):
    assert name == 'scan'
    return ViewSpec('frame')
""",
        encoding="utf-8",
    )
    return tmp_path


def _case_and_path():
    documents = synthetic_objects()
    return documents["ScanPlan"], documents["PathBundle"]


@pytest.fixture(scope="module")
def acp_contract_root(tmp_path_factory):
    # Native integration tests preload ACP; isolate this synthetic checkout.
    saved = {k: v for k, v in sys.modules.items() if k == "acp" or k.startswith("acp.")}
    for name in saved:
        sys.modules.pop(name, None)
    try:
        yield _acp_contract_root(tmp_path_factory.mktemp("acp-contract"))
    finally:
        for name in list(sys.modules):
            if name == "acp" or name.startswith("acp."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


def test_projects_path_through_acp_frame_contract_and_keeps_energy_channels_separate(acp_contract_root):
    plan, path = _case_and_path()
    path = dict(path)
    frames = [dict(frame) for frame in path["frames"]]
    frames[0] = {
        **frames[0],
        "geometry_ref": "RESULT/pes_search/frames/frame_0000.xyz",
        "energies": {
            **frames[0]["energies"],
            "refined_electronic": {
                "value": -14.0,
                "unit": "hartree",
                "method_id": "orca:r2SCAN-3c",
            },
        },
    }
    path["frames"] = frames
    path = seal_document(path)
    proposals = [rank_path_bundle(path, rule="highest_scan_energy")]

    graph = project_path_bundle_to_acp_graph(
        path,
        plan,
        acp_source_root=acp_contract_root,
        proposals=proposals,
    )

    assert graph["job_id"] == path["acp_task_id"]
    assert len(graph["provenance"]["acp_frame_contract_sha256"]) == 64
    assert graph["nodes"][0]["type"] == "frame"
    assert graph["nodes"][0]["id"] == path["frames"][0]["frame_id"]
    assert graph["nodes"][0]["geometry_ref"] == "RESULT/pes_search/frames/frame_0000.xyz"
    assert graph["nodes"][0]["x"] == plan["candidates"][0]["coordinates"][0]["points"][0]
    assert graph["nodes"][0]["metadata"]["atom_map_ids"] == path["atom_map_ids"]
    assert graph["geometries"][graph["nodes"][0]["id"]]["geometry_sha256"] == proposals[0]["selected_frames"][0]["geometry_sha256"]
    series = {entry["id"]: entry for entry in graph["series"]}
    assert series["scan_energy"]["unit"] == "Eh"
    assert series["single_point_energy"]["unit"] == "Eh"
    assert series["scan_energy"]["values"][0] == -14.1
    assert series["single_point_energy"]["values"][0] == -14.0
    assert graph["annotations"][0]["metadata"]["selection_source"] == "ranking"


def test_rejects_proposal_from_old_path_snapshot(acp_contract_root):
    plan, path = _case_and_path()
    proposal = rank_path_bundle(path)
    changed = dict(path)
    changed["frames"] = [dict(frame) for frame in path["frames"]]
    changed["frames"][0]["geometry"] = [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [3.4, 0.0, 0.0]]
    changed = seal_document(changed)

    with pytest.raises(ACPFrameContractError, match="does not match the projected PathBundle"):
        project_path_bundle_to_acp_graph(
            changed, plan, acp_source_root=acp_contract_root, proposals=[proposal]
        )


def test_rejects_proposal_geometry_digest_mismatch(acp_contract_root):
    plan, path = _case_and_path()
    proposal = rank_path_bundle(path)
    proposal["selected_frames"][0]["geometry_sha256"] = "0" * 64
    proposal = seal_document(proposal)

    with pytest.raises(ACPFrameContractError, match="structure digest differs"):
        project_path_bundle_to_acp_graph(
            path, plan, acp_source_root=acp_contract_root, proposals=[proposal]
        )


def test_builds_canonical_acp_pes_profile_with_identity_and_ranking_provenance():
    plan, path = _case_and_path()
    proposals = [rank_path_bundle(path, rule=rule) for rule in
                 ("highest_scan_energy", "internal_scan_peak")]
    profile = path_bundle_to_acp_pes_profile(path, plan, proposals=proposals)

    assert profile["schema_version"] == "pes_profile_v2"
    assert profile["workflow"] == "PESsearch"
    assert profile["status"] == "ready_for_review"
    assert profile["stationary_point_claimed"] is False
    assert profile["scan"]["frames"][0]["pes2ts_frame_id"] == path["frames"][0]["frame_id"]
    assert profile["scan"]["frames"][4]["scan_energy_hartree"] is None
    assert profile["scan"]["frames"][0]["single_point_energy_hartree"] is None
    assert profile["provenance"]["frame_id_map"]["0"] == path["frames"][0]["frame_id"]
    assert {row["rule"] for row in profile["provenance"]["recommendation_sources"].values()} == {
        "highest_scan_energy", "internal_scan_peak"
    }


def test_acp_profile_carries_only_safe_result_geometry_references():
    plan, path = _case_and_path()
    path = dict(path)
    path["frames"] = [dict(frame) for frame in path["frames"]]
    path["frames"][0]["geometry_ref"] = "RESULT/pes_search/frames/frame_0000.xyz"
    path["frames"][1]["geometry_ref"] = "WORK/scan/frame_0001.xyz"
    path = seal_document(path)

    profile = path_bundle_to_acp_pes_profile(path, plan)

    assert profile["scan"]["frames"][0]["geometry_path"] == "RESULT/pes_search/frames/frame_0000.xyz"
    assert profile["scan"]["frames"][1]["geometry_path"] == ""


def test_only_safe_result_relative_geometry_paths_are_retained():
    case, plan = synthetic_objects()["ReactionCase"], synthetic_objects()["ScanPlan"]
    native = {
        "index": 0,
        "target_coordinate": 1.1,
        "geometry_path": "../WORK/secret.xyz",
        "scan_energy_hartree": -10.0,
        "optimization_converged": True,
    }
    bundle = acp_result_to_path_bundle(
        case=case,
        plan=plan,
        execution_id="exec-geometry-ref",
        acp_task_id="job-geometry-ref",
        frames=[native],
        geometry_loader=lambda _: case["reactant"]["geometry"],
    )
    assert bundle["frames"][0]["geometry_ref"] == ""
    native["geometry_path"] = "pes_search/scan/frame_0000.xyz"
    bundle = acp_result_to_path_bundle(
        case=case,
        plan=plan,
        execution_id="exec-geometry-ref-safe",
        acp_task_id="job-geometry-ref-safe",
        frames=[native],
        geometry_loader=lambda _: case["reactant"]["geometry"],
    )
    assert bundle["frames"][0]["geometry_ref"] == "RESULT/pes_search/scan/frame_0000.xyz"
