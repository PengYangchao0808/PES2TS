"""WP-2 action-support registry counterexamples (RL repair plan §5/§14).

Covers the hard gates: source labels never change a support verdict;
environment/reward/driver-order/dimension mismatches are typed rejections;
state and action coverage are separate; the same registry gates the TD
bootstrap and Q execution; unsupported successor actions cancel the
bootstrap with a recorded reason; the legacy source-label gate is only used
when no registry is supplied and is flagged as such.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.generation.control_round import (  # noqa: E402
    ControlRoundError,
    load_control_round_config,
    run_control_collection,
    run_control_execution,
    run_control_freeze,
    run_control_training,
)
from pes2ts_core.pesn.contracts.action_support import (  # noqa: E402
    ACTION_SUPPORT_REGISTRY_SCHEMA,
    ActionSupportContractError,
    validate_action_support_registry,
)
from pes2ts_core.pesn.inference.action_support import (  # noqa: E402
    ActionSupportRegistry,
    standardize_geometry,
)
from pes2ts_core.pesn.offline.action_support import (  # noqa: E402
    build_action_support_registry,
)
from pes2ts_core.pesn.offline.transitions import (  # noqa: E402
    QualifiedTransition,
    load_transition_set,
)
from tests.test_control_round import _build_two_case_inputs  # noqa: E402


def _registry_doc(tmp_path: Path) -> dict:
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
    from pes2ts_core.generation.control_round import build_reward_spec

    load = load_transition_set(
        round_dir, environment_hash=freeze_doc["environment_hash"]
    )
    return build_action_support_registry(
        load.qualified,
        environment_hash=freeze_doc["environment_hash"],
        reward_spec_sha256=build_reward_spec(config).content_hash(),
    )


def _first_entry(registry_doc: dict) -> tuple[dict, dict]:
    group = registry_doc["groups"][0]
    return group, group["entries"][0]


def test_registry_roundtrip_and_built_from_executed_only(tmp_path):
    doc = _registry_doc(tmp_path)
    assert doc["schema_version"] == ACTION_SUPPORT_REGISTRY_SCHEMA
    validate_action_support_registry(doc)
    registry = ActionSupportRegistry(doc)
    assert registry.document is not None
    n_entries = sum(len(g["entries"]) for g in doc["groups"])
    assert n_entries == doc["built_from"]["n_fit_transitions"]
    for group in doc["groups"]:
        for entry in group["entries"]:
            assert isinstance(entry["accepted"], bool)
            assert entry["episode_status"]
            assert len(entry["residual_u"]) == len(group["driver_ids"])
    duplicate = json.loads(json.dumps(doc))
    duplicate["groups"][0]["entries"][0]["transition_id"] = duplicate["groups"][0][
        "entries"
    ][1]["transition_id"]
    with pytest.raises(ActionSupportContractError, match="duplicate transition"):
        validate_action_support_registry(duplicate)


def test_source_label_never_changes_support(tmp_path):
    doc = _registry_doc(tmp_path)
    group, entry = _first_entry(doc)
    relabelled = json.loads(json.dumps(doc))
    relabelled["groups"][0]["entries"][0]["source_label"] = "totally_different_label"
    registry_a = ActionSupportRegistry(doc)
    registry_b = ActionSupportRegistry(relabelled)
    driver_ids = group["driver_ids"]

    def query(registry):
        return registry.query(
            reaction_id=group["reaction_id"],
            driver_ids=driver_ids,
            epsilon=group["epsilon"],
            environment_hash=doc["environment_hash"],
            reward_spec_sha256=doc["reward_spec_sha256"],
            coordinate_spec_sha256=group["coordinate_spec_sha256"],
            geometry=entry["state_geometry_std"],
            context_vector=entry["context_vector"],
            candidate_u=entry["residual_u"],
        )

    verdict_a = query(registry_a)
    verdict_b = query(registry_b)
    assert verdict_a.supported is True
    assert verdict_b.supported == verdict_a.supported
    assert verdict_b.neighbor_count == verdict_a.neighbor_count


def test_spec_and_dimension_mismatches_are_typed(tmp_path):
    doc = _registry_doc(tmp_path)
    registry = ActionSupportRegistry(doc)
    group, entry = _first_entry(doc)

    def query(**overrides):
        params = dict(
            reaction_id=group["reaction_id"],
            driver_ids=group["driver_ids"],
            epsilon=group["epsilon"],
            environment_hash=doc["environment_hash"],
            reward_spec_sha256=doc["reward_spec_sha256"],
            coordinate_spec_sha256=group["coordinate_spec_sha256"],
            geometry=entry["state_geometry_std"],
            context_vector=entry["context_vector"],
            candidate_u=entry["residual_u"],
        )
        params.update(overrides)
        return registry.query(**params)

    assert query(environment_hash="0" * 64).reason == "control_spec_mismatch"
    assert query(reward_spec_sha256="0" * 64).reason == "control_spec_mismatch"
    if len(group["driver_ids"]) >= 2:
        assert query(
            driver_ids=list(reversed(group["driver_ids"]))
        ).reason in ("control_spec_mismatch", "dimension_mismatch")
    assert query(candidate_u=[0.0] * (len(group["driver_ids"]) + 1)
                 ).reason == "dimension_mismatch"


def test_state_and_action_coverage_are_separate(tmp_path):
    doc = _registry_doc(tmp_path)
    registry = ActionSupportRegistry(doc)
    group, entry = _first_entry(doc)

    def query(geometry=None, candidate=None):
        return registry.query(
            reaction_id=group["reaction_id"],
            driver_ids=group["driver_ids"],
            epsilon=group["epsilon"],
            environment_hash=doc["environment_hash"],
            reward_spec_sha256=doc["reward_spec_sha256"],
            coordinate_spec_sha256=group["coordinate_spec_sha256"],
            geometry=geometry if geometry is not None else entry["state_geometry_std"],
            context_vector=entry["context_vector"],
            candidate_u=candidate if candidate is not None else entry["residual_u"],
        )

    assert query().supported is True
    displaced = [list(row) for row in entry["state_geometry_std"]]
    displaced[0] = [x + 0.5 for x in displaced[0]]
    verdict = query(geometry=displaced)
    assert verdict.supported is False
    assert verdict.reason == "state_not_covered"

    n_drivers = len(group["driver_ids"])
    odd = [0.37] * n_drivers
    verdict = query(candidate=odd)
    assert verdict.supported is False
    assert verdict.reason == "action_not_covered"
    assert verdict.nearest_action_distance is not None


def test_failed_actions_provide_q_support_but_not_imitation(tmp_path):
    doc = _registry_doc(tmp_path)
    found_failed = False
    for group in doc["groups"]:
        for entry in group["entries"]:
            if not entry["accepted"]:
                found_failed = True
                registry = ActionSupportRegistry(doc)
                verdict = registry.query(
                    reaction_id=group["reaction_id"],
                    driver_ids=group["driver_ids"],
                    epsilon=group["epsilon"],
                    environment_hash=doc["environment_hash"],
                    reward_spec_sha256=doc["reward_spec_sha256"],
                    coordinate_spec_sha256=group["coordinate_spec_sha256"],
                    geometry=entry["state_geometry_std"],
                    context_vector=entry["context_vector"],
                    candidate_u=entry["residual_u"],
                )
                assert verdict.supported is True
    if not found_failed:
        pytest.skip("fixture produced no failed executed action")


def test_td_bootstrap_cancelled_without_supported_successor_action(tmp_path):
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
    from pes2ts_core.generation.control_round import build_reward_spec

    load = load_transition_set(
        round_dir, environment_hash=freeze_doc["environment_hash"]
    )
    doc = build_action_support_registry(
        load.qualified,
        environment_hash=freeze_doc["environment_hash"],
        reward_spec_sha256=build_reward_spec(config).content_hash(),
    )
    registry = ActionSupportRegistry(doc)
    group = doc["groups"][0]
    entry = group["entries"][0]
    source = next(
        t for t in load.qualified
        if t.doc.get("transition_id") == entry["transition_id"]
    )
    transition = QualifiedTransition(
        doc={
            "transition_id": "support-test-d0",
            "reaction_id": group["reaction_id"],
            "decision_index": 0,
            "action": {
                "driver_ids": group["driver_ids"],
                "epsilon": group["epsilon"],
                "residual_u": entry["residual_u"],
                "source": "rule_zero",
            },
            "environment_hash": doc["environment_hash"],
            "environment_id": "E_native",
            "state_after_geometry": entry["state_geometry_std"],
            "bootstrap_mask": True,
        },
        case=source.case,
    )
    from pes2ts_core.pesn.offline.control_trainer import (
        ControlTrainer,
        ControlTrainSettings,
    )

    settings = ControlTrainSettings(training_mode="cql", max_epochs=1)
    trainer = ControlTrainer(
        settings=settings, run_record=None,
        config_sha256="0" * 64, transitions_digest="0" * 64,
        support_registry=registry,
    )
    n_drivers = len(group["driver_ids"])
    unsupported = [[0.37] * n_drivers]
    mask = trainer._successor_support_mask(
        transition, tuple(entry["context_vector"]), unsupported
    )
    assert mask == [False]
    assert trainer.bootstrap_unsupported_reasons.get("action_not_covered") == 1

    supported = [list(entry["residual_u"])]
    mask = trainer._successor_support_mask(
        transition, tuple(entry["context_vector"]), supported
    )
    assert mask == [True]

    original_mask_value = transition.doc["bootstrap_mask"]
    assert original_mask_value is True


def test_trainer_summary_reports_bootstrap_gate(tmp_path):
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
    summary = run_control_training(
        round_dir=round_dir, config=config, selection_path=selection_path
    )
    assert summary["bootstrap_support"]["gate"] == "registry_v1"
    assert summary["bootstrap_support"]["n_cancelled_no_supported_successor_action"] >= 0
    registry_path = Path(summary["action_support_registry_path"])
    assert registry_path.is_file()
    registry_doc = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_action_support_registry(registry_doc)


def test_execution_uses_registry_gate_when_provided(tmp_path):
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
    train = run_control_training(
        round_dir=round_dir, config=config, selection_path=selection_path
    )
    freeze = run_control_freeze(
        round_dir=round_dir, config=config, selection_path=selection_path
    )
    bundle_dir = Path(freeze["bundle_path"])
    registry_path = Path(train["action_support_registry_path"])

    executed = run_control_execution(
        round_dir=round_dir, manifest_path=manifest_path, legacy_root=legacy_root,
        selection_path=selection_path, config=config, bundle_dir=bundle_dir,
        transport=transport, support_registry_path=registry_path,
        resume=False,
    )
    gate = executed["support_gate"]
    assert gate["kind"] == "registry_v1_bound"
    assert gate["legacy_support"] is False
    assert gate["registry_sha256"]
    assert gate["registry_source"] == "bundle_binding"

    bound = run_control_execution(
        round_dir=round_dir, manifest_path=manifest_path, legacy_root=legacy_root,
        selection_path=selection_path, config=config, bundle_dir=bundle_dir,
        transport=transport, resume=False,
    )
    assert bound["support_gate"]["kind"] == "registry_v1_bound"
    assert bound["support_gate"]["legacy_support"] is False


def test_supported_high_score_nonzero_action_is_selected(tmp_path):
    """Hard gate §5.2: a constructible supported high-score non-zero action
    must win the Q selection over the zero baseline."""
    from pes2ts_core.generation.planning.policy_adapter import (
        CandidatePolicyConfig,
        QGreedyPolicy,
    )
    from tests.test_control_round import build_spec_like, build_test_control_like

    control = build_test_control_like()
    spec = build_spec_like(control)
    geometry = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]

    reaction_id = "RXN_Q_SELECTION_TEST"
    driver_ids = [str(d) for d in spec.driver_ids]
    epsilon = [float(e) for e in spec.epsilon]
    from pes2ts_core.pesn.inference.action_support import standardize_geometry

    std = standardize_geometry(geometry)
    context = [0.5, 0.55, 1.0, 1.0, 0.0, 0.0, 0.0]
    supported_u = [1.0] * len(driver_ids)
    doc = {
        "schema_version": "pesn_action_support_registry_v1",
        "environment_hash": "e" * 64,
        "reward_spec_sha256": "r" * 64,
        "built_from": {"fit_transition_ids": ["t0"], "n_fit_transitions": 1},
        "calibration": {
            "action_tolerance_std": 0.25,
            "state_tolerance_angstrom": 1e-3,
            "context_tolerance": 1e-6,
            "policy": "test",
        },
        "groups": [{
            "reaction_id": reaction_id,
            "driver_ids": driver_ids,
            "epsilon": epsilon,
            "coordinate_spec_sha256": "c" * 64,
            "n_states": 1,
            "entries": [{
                "transition_id": "t0",
                "state_geometry_std": [
                    [round(float(x), 9) for x in row] for row in std
                ],
                "context_vector": context,
                "residual_u": supported_u,
                "accepted": True,
                "episode_status": "ongoing",
                "source_label": "bounded_exploration",
            }],
        }],
    }
    registry = ActionSupportRegistry(doc)

    class State:
        pass

    state = State()
    state.control = control
    state.spec = spec
    state.target_lambda = 0.5

    def support_gate(_state, candidates, _sources):
        return [
            registry.query(
                reaction_id=reaction_id,
                driver_ids=driver_ids,
                epsilon=epsilon,
                environment_hash="e" * 64,
                reward_spec_sha256="r" * 64,
                coordinate_spec_sha256="c" * 64,
                geometry=geometry,
                context_vector=context,
                candidate_u=list(candidate),
            ).supported
            for candidate in candidates
        ]

    def score(_state, candidates):
        values = []
        for candidate in candidates:
            if support_gate(_state, [candidate], [])[0]:
                is_nonzero = any(abs(u) > 0 for u in candidate)
                values.append(1.0 if is_nonzero else 0.5)
            else:
                values.append(None)
        return values

    policy = QGreedyPolicy(
        CandidatePolicyConfig(), score, support_fn=support_gate
    )
    decision = policy.decide(state)
    executed = decision.candidates[decision.executed_candidate_index]
    assert any(abs(u) > 0 for u in executed), (
        "a supported, highest-scored non-zero action must be selected"
    )


def test_execution_refuses_foreign_registry(tmp_path):
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
    train = run_control_training(
        round_dir=round_dir, config=config, selection_path=selection_path
    )
    freeze = run_control_freeze(
        round_dir=round_dir, config=config, selection_path=selection_path
    )
    registry_path = Path(train["action_support_registry_path"])
    doc = json.loads(registry_path.read_text(encoding="utf-8"))
    doc["environment_hash"] = "0" * 64
    foreign = tmp_path / "foreign_registry.json"
    foreign.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(
        ControlRoundError, match="differs from the registry bound"
    ):
        run_control_execution(
            round_dir=round_dir, manifest_path=manifest_path, legacy_root=legacy_root,
            selection_path=selection_path, config=config,
            bundle_dir=Path(freeze["bundle_path"]), transport=transport,
            support_registry_path=foreign,
        )
