"""Repair batch 2 counterexamples: review findings F1-F4 (2026-10-08).

F1 — the support query must filter geometry AND context jointly per entry;
     action coverage is computed only over the jointly compatible rows, and
     the raw-nearest vs compatible-nearest state distances stay distinct.
F2 — the TD support registry consumes the selection-whitelisted transition
     list; an unauthorized record in the store can never enter TD support.
F3 — freeze binds the training registry into the bundle; execution uses the
     bound registry automatically, rejects a differing explicit registry,
     and only unbound legacy packages fall back to the source-label gate.
F4 — a missing group/split evidence source yields a typed unknown verdict
     with inherit=False, never a passing integrity flag.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.generation.control_round import (  # noqa: E402
    ControlRoundError,
    _load_td_support_registry,
    load_control_round_config,
    run_control_collection,
    run_control_execution,
    run_control_freeze,
    run_control_training,
)
from pes2ts_core.pesn.bundle.reader import read_bundle_manifest  # noqa: E402
from pes2ts_core.pesn.contracts.control import (  # noqa: E402
    validate_support_registry_binding,
)
from pes2ts_core.pesn.inference.action_support import (  # noqa: E402
    ActionSupportRegistry,
)
from pes2ts_core.pesn.offline.transitions import (  # noqa: E402
    append_transitions,
    load_transition_set,
)
from tests.test_control_round import _build_two_case_inputs  # noqa: E402


def _collect_fixture_round(tmp_path: Path):
    (manifest_path, legacy_root, selection_path, config_path) = (
        _build_two_case_inputs(tmp_path, drop_second_plan=False)
    )
    config = load_control_round_config(config_path)
    round_dir = tmp_path / "rounds" / "r1"
    from tests.test_control_step import make_transport

    transport, _ = make_transport()
    run_control_collection(
        round_dir=round_dir, manifest_path=manifest_path, legacy_root=legacy_root,
        selection_path=selection_path, config=config, transport=transport,
    )
    freeze_doc = json.loads(
        (round_dir / "environment_freeze.json").read_text(encoding="utf-8")
    )
    return {
        "manifest_path": manifest_path,
        "legacy_root": legacy_root,
        "selection_path": selection_path,
        "config": config,
        "round_dir": round_dir,
        "transport": transport,
        "freeze_doc": freeze_doc,
    }


def _two_context_registry_doc(*, order: str = "AB") -> dict:
    geometry = [[-1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
    entry_a = {
        "transition_id": "t-a",
        "state_geometry_std": [list(row) for row in geometry],
        "context_vector": [0.0] * 7,
        "residual_u": [0.0, 0.0],
        "accepted": True,
        "episode_status": "ongoing",
        "source_label": "rule_zero",
    }
    entry_b = {
        "transition_id": "t-b",
        "state_geometry_std": [list(row) for row in geometry],
        "context_vector": [1.0] * 7,
        "residual_u": [1.0, 1.0],
        "accepted": True,
        "episode_status": "ongoing",
        "source_label": "bounded_exploration",
    }
    entries = [entry_a, entry_b] if order == "AB" else [entry_b, entry_a]
    return {
        "schema_version": "pesn_action_support_registry_v1",
        "environment_hash": "e" * 64,
        "reward_spec_sha256": "r" * 64,
        "built_from": {
            "fit_transition_ids": ["t-a", "t-b"],
            "n_fit_transitions": 2,
        },
        "calibration": {
            "action_tolerance_std": 0.25,
            "state_tolerance_angstrom": 1e-3,
            "context_tolerance": 1e-6,
            "policy": "f1_counterexample",
        },
        "groups": [{
            "reaction_id": "RXN_F1",
            "driver_ids": ["d0", "d1"],
            "epsilon": [1.0, 1.0],
            "coordinate_spec_sha256": "c" * 64,
            "n_states": 2,
            "entries": entries,
        }],
    }


def _f1_query(registry: ActionSupportRegistry, context, candidate):
    return registry.query(
        reaction_id="RXN_F1",
        driver_ids=["d0", "d1"],
        epsilon=[1.0, 1.0],
        environment_hash="e" * 64,
        reward_spec_sha256="r" * 64,
        coordinate_spec_sha256="c" * 64,
        geometry=[[-1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        context_vector=context,
        candidate_u=candidate,
    )


class TestF1JointGeometryContextMask:
    def test_action_only_present_under_incompatible_context_is_rejected(self):
        registry = ActionSupportRegistry(_two_context_registry_doc())
        verdict = _f1_query(registry, [0.0] * 7, [1.0, 1.0])
        assert verdict.supported is False
        assert verdict.reason == "action_not_covered"

    def test_compatible_context_entry_is_not_wrongly_rejected(self):
        registry = ActionSupportRegistry(_two_context_registry_doc())
        verdict = _f1_query(registry, [1.0] * 7, [1.0, 1.0])
        assert verdict.supported is True
        assert verdict.neighbor_count == 1

    @pytest.mark.parametrize("order", ["AB", "BA"])
    def test_entry_order_does_not_change_verdicts(self, order):
        registry = ActionSupportRegistry(_two_context_registry_doc(order=order))
        a_ctx_a_action = _f1_query(registry, [0.0] * 7, [0.0, 0.0])
        a_ctx_b_action = _f1_query(registry, [0.0] * 7, [1.0, 1.0])
        b_ctx_b_action = _f1_query(registry, [1.0] * 7, [1.0, 1.0])
        b_ctx_a_action = _f1_query(registry, [1.0] * 7, [0.0, 0.0])
        assert a_ctx_a_action.supported is True
        assert a_ctx_b_action.supported is False
        assert b_ctx_b_action.supported is True
        assert b_ctx_a_action.supported is False

    def test_nearest_distances_distinguish_raw_and_compatible(self):
        doc = _two_context_registry_doc()
        far = json.loads(json.dumps(doc))
        far["groups"][0]["entries"][1]["state_geometry_std"][0] = [0.4, 0.0, 0.0]
        registry = ActionSupportRegistry(far)
        verdict = _f1_query(registry, [1.0] * 7, [1.0, 1.0])
        assert verdict.supported is False
        assert verdict.reason == "state_not_covered"
        assert verdict.nearest_state_distance == pytest.approx(0.0, abs=1e-9)
        assert verdict.nearest_compatible_state_distance is None

        verdict = _f1_query(registry, [0.0] * 7, [0.0, 0.0])
        assert verdict.supported is True
        assert verdict.nearest_compatible_state_distance is not None


class TestF2TDRegistryTrainingUseWhitelist:
    def test_unauthorized_record_never_enters_td_registry(self, tmp_path):
        fixture = _collect_fixture_round(tmp_path)
        round_dir = fixture["round_dir"]
        selection = json.loads(
            Path(fixture["selection_path"]).read_text(encoding="utf-8")
        )
        allowed = list(selection["reaction_ids"])
        store = round_dir / "transitions" / "control_transitions.jsonl"
        first = json.loads(store.read_text(encoding="utf-8").splitlines()[0])
        injected = json.loads(json.dumps(first))
        injected["reaction_id"] = "RXN_UNAUTHORIZED"
        injected["transition_id"] = "RXN_UNAUTHORIZED-d000"
        append_transitions(store, [injected])

        from pes2ts_core.pesn.offline.control_trainer import ControlTrainSettings

        settings = ControlTrainSettings(training_mode="cql", max_epochs=1)
        whitelisted = load_transition_set(
            round_dir,
            environment_hash=fixture["freeze_doc"]["environment_hash"],
            allowed_reactions=allowed,
        ).qualified
        assert all(t.reaction_id in set(allowed) for t in whitelisted)

        registry = _load_td_support_registry(
            round_dir,
            fixture["config"],
            fixture["freeze_doc"],
            settings,
            whitelisted,
        )
        assert registry is not None
        fit_ids = set(
            registry.document["built_from"]["fit_transition_ids"]
        )
        assert "RXN_UNAUTHORIZED-d000" not in fit_ids
        assert all(
            not tid.startswith("RXN_UNAUTHORIZED") for tid in fit_ids
        )

    def test_training_summary_keeps_td_and_export_registry_identical(
        self, tmp_path
    ):
        fixture = _collect_fixture_round(tmp_path)
        summary = run_control_training(
            round_dir=fixture["round_dir"],
            config=fixture["config"],
            selection_path=fixture["selection_path"],
        )
        assert summary["td_support_registry_sha256"] == (
            summary["action_support_registry_sha256"]
        )
        assert summary["support_registry_td_export_consistent"] is True


class TestF3RegistryBoundIntoBundle:
    def _trained_and_frozen(self, tmp_path):
        fixture = _collect_fixture_round(tmp_path)
        train = run_control_training(
            round_dir=fixture["round_dir"],
            config=fixture["config"],
            selection_path=fixture["selection_path"],
        )
        freeze = run_control_freeze(
            round_dir=fixture["round_dir"],
            config=fixture["config"],
            selection_path=fixture["selection_path"],
        )
        return fixture, train, freeze

    def test_freeze_binds_registry_and_reader_accepts(self, tmp_path):
        _fixture, train, freeze = self._trained_and_frozen(tmp_path)
        bundle_dir = Path(freeze["bundle_path"])
        manifest, _root = read_bundle_manifest(bundle_dir)
        binding = manifest["support_registry"]
        validate_support_registry_binding(binding)
        assert binding["sha256"] == train["action_support_registry_sha256"]
        registry_file = bundle_dir / "action_support_registry.json"
        assert registry_file.is_file()
        import hashlib

        doc = json.loads(registry_file.read_text(encoding="utf-8"))
        content_sha = hashlib.sha256(
            json.dumps(doc, sort_keys=True).encode("utf-8")
        ).hexdigest()
        assert content_sha == binding["sha256"]
        checksums = json.loads(
            (bundle_dir / "checksums.json").read_text(encoding="utf-8")
        )
        assert checksums["action_support_registry.json"] == (
            hashlib.sha256(registry_file.read_bytes()).hexdigest()
        )
        assert freeze["support_registry"]["bound"] is True

    def test_execute_uses_bound_registry_without_cli_flag(self, tmp_path):
        fixture, _train, freeze = self._trained_and_frozen(tmp_path)
        executed = run_control_execution(
            round_dir=fixture["round_dir"],
            manifest_path=fixture["manifest_path"],
            legacy_root=fixture["legacy_root"],
            selection_path=fixture["selection_path"],
            config=fixture["config"],
            bundle_dir=Path(freeze["bundle_path"]),
            transport=fixture["transport"],
            resume=False,
        )
        gate = executed["support_gate"]
        assert gate["kind"] == "registry_v1_bound"
        assert gate["legacy_support"] is False
        assert gate["registry_source"] == "bundle_binding"

    def test_execute_rejects_differing_explicit_registry(self, tmp_path):
        fixture, train, freeze = self._trained_and_frozen(tmp_path)
        foreign = tmp_path / "foreign_registry.json"
        doc = json.loads(
            Path(train["action_support_registry_path"]).read_text(
                encoding="utf-8"
            )
        )
        doc["calibration"]["action_tolerance_std"] = 0.5
        foreign.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(
            ControlRoundError, match="differs from the registry bound"
        ):
            run_control_execution(
                round_dir=fixture["round_dir"],
                manifest_path=fixture["manifest_path"],
                legacy_root=fixture["legacy_root"],
                selection_path=fixture["selection_path"],
                config=fixture["config"],
                bundle_dir=Path(freeze["bundle_path"]),
                transport=fixture["transport"],
                support_registry_path=foreign,
            )

    def test_tampered_bound_registry_file_is_rejected(self, tmp_path):
        fixture, _train, freeze = self._trained_and_frozen(tmp_path)
        bundle_dir = Path(freeze["bundle_path"])
        registry_file = bundle_dir / "action_support_registry.json"
        doc = json.loads(registry_file.read_text(encoding="utf-8"))
        doc["calibration"]["action_tolerance_std"] = 0.99
        registry_file.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(Exception, match="(?i)checksum|hash mismatch"):
            run_control_execution(
                round_dir=fixture["round_dir"],
                manifest_path=fixture["manifest_path"],
                legacy_root=fixture["legacy_root"],
                selection_path=fixture["selection_path"],
                config=fixture["config"],
                bundle_dir=bundle_dir,
                transport=fixture["transport"],
                resume=False,
            )

    def test_legacy_unbound_bundle_still_uses_source_gate(self, tmp_path):
        fixture, train, _freeze = self._trained_and_frozen(tmp_path)
        from pes2ts_core.generation.control_round import build_reward_spec
        from pes2ts_core.pesn.offline.publishing.control_export import (
            export_control_bundle,
        )

        run_dir = (
            fixture["round_dir"] / "runs" / train["run_id"]
        )
        frozen_manifest = json.loads(
            (Path(_freeze["bundle_path"]) / "model_bundle.json").read_text(
                encoding="utf-8"
            )
        )
        legacy_bundle = tmp_path / "artifacts" / "legacy-unbound"
        export_control_bundle(
            run_dir=run_dir,
            output_dir=legacy_bundle,
            model_id="legacy-unbound-test",
            metrics={
                "status": "research_only_first_round",
                "training_mode": train["training_mode"],
                "environment_hash": fixture["freeze_doc"]["environment_hash"],
                "supported_action_sources": ["rule_zero"],
            },
            run_summary={
                "run_id": train["run_id"],
                "training_mode": train["training_mode"],
                "effective_config_sha256": train["effective_config_sha256"],
                "transitions_digest": train["transitions_digest"],
            },
            model_config=dict(frozen_manifest["model_config"]),
            control_spec_sha256="0" * 64,
            reward_spec_sha256=build_reward_spec(
                fixture["config"]
            ).content_hash(),
            environment_id="E_native",
            support_registry_doc=None,
        )
        executed = run_control_execution(
            round_dir=fixture["round_dir"],
            manifest_path=fixture["manifest_path"],
            legacy_root=fixture["legacy_root"],
            selection_path=fixture["selection_path"],
            config=fixture["config"],
            bundle_dir=legacy_bundle,
            transport=fixture["transport"],
            resume=False,
        )
        gate = executed["support_gate"]
        assert gate["kind"] == "legacy_source_labels"
        assert gate["legacy_support"] is True


class TestF4MissingEvidenceIsTypedUnknown:
    def test_missing_sources_return_unknown_and_block_inheritance(
        self, tmp_path, monkeypatch
    ):
        import scripts.audit_rl_round0 as audit

        monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
        reference = audit._build_split_reference(["RXN_A", "RXN_B"])
        assert reference["integrity_status"] == (
            "unknown_missing_evidence_sources"
        )
        assert reference["group_integrity_ok"] is False
        assert reference["inherited_fit_dev_split"]["inherit"] is False
        assert sorted(reference["unavailable_sources"]) == [
            "duplicate_groups: FileNotFoundError",
            "split_assignment: FileNotFoundError",
        ]
        assert reference["source_status"]["duplicate_groups"]["available"] is False
        assert reference["source_status"]["split_assignment"]["available"] is False
