"""Multi-episode successor identity (IRC20k/RL2000 plan §5, review WP-0 §3.1.5).

Same reaction + same decision_index from different episodes/branches must
never chain successors; the episode-index sidecar is the unique identity;
incompatible geometric links are recorded, not bootstrapped; legacy stores
without an index keep the historical same-reaction semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.generation.control_round import (  # noqa: E402
    load_control_round_config,
    run_control_collection,
)
from pes2ts_core.pesn.offline.control_trainer import (  # noqa: E402
    _successor_context_index,
)
from pes2ts_core.pesn.offline.transitions import (  # noqa: E402
    QualifiedTransition,
    TransitionStoreError,
    count_episodes_for_reaction,
    load_episode_transition_map,
    load_transition_set,
    register_episode,
)
from tests.test_control_round import _build_two_case_inputs  # noqa: E402


def _collect_round(tmp_path: Path):
    (manifest_path, legacy_root, selection_path, config_path) = (
        _build_two_case_inputs(tmp_path, drop_second_plan=False)
    )
    config = load_control_round_config(config_path)
    round_dir = tmp_path / "rounds" / "r1"
    from tests.test_control_step import make_transport

    transport, _ = make_transport()
    summary = run_control_collection(
        round_dir=round_dir, manifest_path=manifest_path, legacy_root=legacy_root,
        selection_path=selection_path, config=config, transport=transport,
    )
    return round_dir, summary


def _episode_docs(round_dir: Path, reaction_id: str) -> list[dict]:
    from pes2ts_core.pesn.offline.transitions import iter_transition_docs

    return [
        doc
        for doc in iter_transition_docs(
            round_dir / "transitions" / "control_transitions.jsonl"
        )
        if doc["reaction_id"] == reaction_id
    ]


class TestEpisodeIndexSidecar:
    def test_collect_registers_one_episode_per_case(self, tmp_path):
        round_dir, summary = _collect_round(tmp_path)
        executed = [r for r in summary["cases"] if not r.get("resumed")]
        assert executed, "fixture produced no executed cases"
        index_path = round_dir / "transitions" / "episode_index.json"
        assert index_path.is_file()
        document = json.loads(index_path.read_text(encoding="utf-8"))
        assert document["schema_version"] == "pesn_control_episode_index_v1"
        for row in executed:
            assert row["episode_id"] == (
                f"{round_dir.name}:{row['reaction_id']}:e000"
            )
        for entry in document["episodes"]:
            ids = entry["transition_ids"]
            assert ids and all(f"-e000-d" in tid for tid in ids)

    def test_second_episode_same_reaction_gets_new_identity(self, tmp_path):
        round_dir, first = _collect_round(tmp_path)
        reaction_id = first["cases"][0]["reaction_id"]
        assert count_episodes_for_reaction(round_dir, reaction_id) == 1
        finalized_dir = Path(round_dir) / "cases" / reaction_id / "control"
        for marker in finalized_dir.glob("control_scan_result.json"):
            marker.unlink()
        (manifest_path, legacy_root, selection_path, config_path) = (
            _build_two_case_inputs(tmp_path, drop_second_plan=False)
        )
        from tests.test_control_step import make_transport

        transport, _ = make_transport()
        second = run_control_collection(
            round_dir=round_dir, manifest_path=manifest_path,
            legacy_root=legacy_root, selection_path=selection_path,
            config=load_control_round_config(config_path), transport=transport,
        )
        assert count_episodes_for_reaction(round_dir, reaction_id) == 2
        old_ids = {
            d["transition_id"] for d in _episode_docs(round_dir, reaction_id)
            if "-e000-" in d["transition_id"]
        }
        new_ids = {
            d["transition_id"] for d in _episode_docs(round_dir, reaction_id)
            if "-e001-" in d["transition_id"]
        }
        assert old_ids and new_ids
        assert not (old_ids & new_ids)

    def test_register_episode_is_idempotent_and_rejects_sharing(self, tmp_path):
        round_dir, summary = _collect_round(tmp_path)
        reaction_id = summary["cases"][0]["reaction_id"]
        docs = [
            d
            for d in _episode_docs(round_dir, reaction_id)
            if "-e000-" in d["transition_id"]
        ]
        entry = register_episode(
            round_dir,
            episode_id=f"{round_dir.name}:{reaction_id}:e000",
            round_id=round_dir.name,
            reaction_id=reaction_id,
            transition_docs=docs,
            policy_id="p", policy_hash="h",
        )
        assert entry["episode_id"].endswith(":e000")
        with pytest.raises(TransitionStoreError, match="never share transitions"):
            register_episode(
                round_dir,
                episode_id=f"{round_dir.name}:{reaction_id}:e002",
                round_id=round_dir.name,
                reaction_id=reaction_id,
                transition_docs=docs,
                policy_id="p", policy_hash="h",
            )

    def test_non_contiguous_decisions_are_refused(self, tmp_path):
        round_dir, summary = _collect_round(tmp_path)
        reaction_id = summary["cases"][0]["reaction_id"]
        docs = sorted(
            (d for d in _episode_docs(round_dir, reaction_id)
             if "-e000-" in d["transition_id"]),
            key=lambda d: d["decision_index"],
        )
        broken = [dict(docs[0]), dict(docs[-1])]
        broken[1]["transition_id"] = broken[1]["transition_id"] + "x"
        with pytest.raises(TransitionStoreError, match="not contiguous"):
            register_episode(
                round_dir,
                episode_id=f"{round_dir.name}:{reaction_id}:e009",
                round_id=round_dir.name,
                reaction_id=reaction_id,
                transition_docs=broken,
                policy_id="p", policy_hash="h",
            )


class TestEpisodeAwareSuccessorLinking:
    def _qualified(self, round_dir: Path, reaction_id: str):
        freeze = json.loads(
            (round_dir / "environment_freeze.json").read_text(encoding="utf-8")
        )
        load = load_transition_set(
            round_dir, environment_hash=freeze["environment_hash"]
        )
        return [t for t in load.qualified if t.reaction_id == reaction_id]

    def test_cross_episode_successor_never_chains(self, tmp_path):
        round_dir, first = _collect_round(tmp_path)
        reaction_id = first["cases"][0]["reaction_id"]
        finalized = round_dir / "cases" / reaction_id / "control"
        for marker in finalized.glob("control_scan_result.json"):
            marker.unlink()
        (manifest_path, legacy_root, selection_path, config_path) = (
            _build_two_case_inputs(tmp_path, drop_second_plan=False)
        )
        from tests.test_control_step import make_transport

        transport, _ = make_transport()
        run_control_collection(
            round_dir=round_dir, manifest_path=manifest_path,
            legacy_root=legacy_root, selection_path=selection_path,
            config=load_control_round_config(config_path), transport=transport,
        )
        qualified = self._qualified(round_dir, reaction_id)
        by_episode: dict[str, list[QualifiedTransition]] = {}
        for t in qualified:
            assert t.episode is not None
            by_episode.setdefault(t.episode_id, []).append(t)
        assert len(by_episode) == 2
        index = _successor_context_index(qualified)
        for episode_id, members in by_episode.items():
            members.sort(key=lambda t: t.doc["decision_index"])
            last = members[-1]
            assert (episode_id, last.doc["decision_index"] + 1) not in index

    def test_incompatible_geometric_link_is_not_registered(self, tmp_path):
        round_dir, summary = _collect_round(tmp_path)
        reaction_id = summary["cases"][0]["reaction_id"]
        docs = sorted(
            (d for d in _episode_docs(round_dir, reaction_id)
             if "-e000-" in d["transition_id"]),
            key=lambda d: d["decision_index"],
        )
        assert len(docs) >= 2
        docs[0]["state_after_geometry"] = [
            [float(x) + 1.0 for x in row] for row in docs[0]["state_after_geometry"]
        ]
        for doc in docs:
            doc["transition_id"] = doc["transition_id"].replace("-e000-", "-e004-")
        register_episode(
            round_dir,
            episode_id=f"{round_dir.name}:{reaction_id}:e004",
            round_id=round_dir.name,
            reaction_id=reaction_id,
            transition_docs=docs,
            policy_id="p", policy_hash="h",
        )
        mapping = load_episode_transition_map(round_dir)
        first_new = next(
            meta
            for tid, meta in mapping.items()
            if "-e004-" in tid and meta["position"] == 1
        )
        assert first_new["incoming_link_compatible"] is False

    def test_legacy_store_without_index_keeps_reaction_linking(self, tmp_path):
        round_dir, summary = _collect_round(tmp_path)
        index_path = round_dir / "transitions" / "episode_index.json"
        index_path.unlink()
        reaction_id = summary["cases"][0]["reaction_id"]
        qualified = self._qualified(round_dir, reaction_id)
        assert all(t.episode is None for t in qualified)
        index = _successor_context_index(qualified)
        for t in qualified:
            assert t.episode_successor_key_base == t.reaction_id
            if not t.terminal:
                key = (t.reaction_id, int(t.doc["decision_index"]) + 1)
                assert key in index
