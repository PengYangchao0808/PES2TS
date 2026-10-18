"""Control contract tests: ADR-0017 schemas, legality, reward, capabilities."""

from __future__ import annotations

import json

import pytest

from pes2ts_core.pesn.contracts.control import (
    ACTION_SOURCE_MODEL_RESIDUAL,
    ACTION_SOURCE_RULE_ZERO,
    CONTROL_TRANSITION_SCHEMA_VERSION,
    CandidateQEstimate,
    ControlAction,
    ControlContractError,
    ControlContext,
    ControlInput,
    ControlPrediction,
    MacroCost,
    PolicyRewardSpec,
    ResidualControlSpec,
    TERMINAL_COMPLETE,
    control_capability_map,
    validate_control_bundle_manifest,
    validate_control_capability_map,
)


def make_spec(**overrides) -> ResidualControlSpec:
    base = dict(
        schema_version="pesn_residual_control_spec_v1",
        driver_ids=("B_1_2",),
        epsilon=(0.1,),
        allowed_sign=(0,),
        min_target=(1.4,),
        max_target=(2.6,),
    )
    base.update(overrides)
    return ResidualControlSpec(**base)


def make_action(**overrides) -> ControlAction:
    base = dict(
        schema_version="pes2ts_control_action_v2",
        action_id="a1",
        driver_ids=("B_1_2",),
        base_targets=(2.0,),
        residual_u=(0.5,),
        epsilon=(0.1,),
        command_targets=(2.05,),
        source=ACTION_SOURCE_MODEL_RESIDUAL,
        policy_id="p1",
    )
    base.update(overrides)
    return ControlAction(**base)


class TestResidualControlSpec:
    def test_valid_spec_passes_and_hashes(self):
        spec = make_spec()
        spec.validate()
        assert len(spec.content_hash()) == 64

    def test_zero_epsilon_rejected(self):
        with pytest.raises(ControlContractError, match="epsilon"):
            make_spec(epsilon=(0.0,)).validate()

    def test_illegal_residual_by_range(self):
        spec = make_spec()
        legal, reason = spec.residual_is_legal((1.0,), (2.55,))
        assert not legal and "range" in reason

    def test_illegal_residual_by_sign(self):
        spec = make_spec(allowed_sign=(1,))
        legal, _ = spec.residual_is_legal((-1.0,), (2.0,))
        assert not legal
        legal, _ = spec.residual_is_legal((-1e-12,), (2.0,))
        assert legal

    def test_command_targets_round_trip(self):
        spec = make_spec()
        command = spec.command_targets((2.0,), (0.5,))
        assert command == pytest.approx((2.05,))


class TestControlContextAndInput:
    def test_context_bounds(self):
        ControlContext(
            nominal_lambda=0.5, next_target_lambda=0.75,
            remaining_wall_fraction=0.4, remaining_request_fraction=0.9,
            recent_rejections=1, rescue_quota_remaining=0,
            candidate_already_awarded=False,
        ).validate()

    def test_context_rejects_negative_memory(self):
        with pytest.raises(ControlContractError, match="negative"):
            ControlContext(
                nominal_lambda=0.5, next_target_lambda=0.75,
                remaining_wall_fraction=0.4, remaining_request_fraction=0.9,
                recent_rejections=-1, rescue_quota_remaining=0,
                candidate_already_awarded=False,
            ).validate()

    def test_control_input_requires_parallel_targets(self):
        class _FakeBase:
            input_hash = "x" * 64

        with pytest.raises(ControlContractError, match="parallel"):
            ControlInput(
                schema_version="pesn_control_input_v1",
                base_input=_FakeBase(),
                context=ControlContext(0.5, 0.75, 0.4, 0.9, 0, 0, False),
                spec=make_spec(driver_ids=("a", "b"), epsilon=(0.1, 0.1),
                               allowed_sign=(0, 0), min_target=(1.0, 1.0),
                               max_target=(2.0, 2.0)),
                base_targets=(1.5,),
            ).validate()


class TestPrediction:
    def test_prediction_validation(self):
        prediction = ControlPrediction(
            schema_version="pesn_control_prediction_v1",
            input_hash="h" * 64, model_id="m", model_hash="x" * 64,
            feature_hash="f" * 64,
            residual=(0.5,), residual_available=True, residual_reason=None,
            q_proxy=(CandidateQEstimate(True, 1.5), CandidateQEstimate(False, None, "no_action_support")),
        )
        prediction.validate()

    def test_residual_out_of_bounds_rejected(self):
        with pytest.raises(ControlContractError, match="\\|u\\|>1"):
            ControlPrediction(
                schema_version="pesn_control_prediction_v1",
                input_hash="h" * 64, model_id="m", model_hash="x" * 64,
                feature_hash="f" * 64,
                residual=(1.5,), residual_available=True, residual_reason=None,
                q_proxy=(),
            ).validate()


class TestAction:
    def test_action_validates_and_hashes(self):
        action = make_action()
        action.validate()
        assert len(action.content_hash()) == 64

    def test_action_command_inconsistency_rejected(self):
        with pytest.raises(ControlContractError, match="inconsistent"):
            make_action(command_targets=(2.5,)).validate()

    def test_action_out_of_bounds_residual_rejected(self):
        with pytest.raises(ControlContractError, match="\\|u\\|>1"):
            make_action(residual_u=(1.4,), command_targets=(2.14,)).validate()


class TestRewardSpec:
    def test_step_reward_terms(self):
        spec = PolicyRewardSpec(
            schema_version="pes2ts_policy_reward_spec_v2",
            candidate_bonus=1.0, cost_reference_seconds=100.0,
            hard_failure_penalty=0.25, hard_failure_codes=frozenset(
                {"STEP_LIMIT:SUBDIVISIONS_EXHAUSTED"}
            ),
        )
        cost = MacroCost(wall_seconds=10.0, host_requests=1,
                         n_optimization_cycles_logged=1, native_wall_seconds=10.0)
        assert spec.step_reward(macro_cost=cost, candidate_awarded=False,
                                hard_failure=False) == pytest.approx(-0.1)
        assert spec.step_reward(macro_cost=cost, candidate_awarded=True,
                                hard_failure=False) == pytest.approx(0.9)
        assert spec.step_reward(macro_cost=cost, candidate_awarded=False,
                                hard_failure=True) == pytest.approx(-0.35)

    def test_gamma_must_stay_one(self):
        with pytest.raises(ControlContractError, match="gamma"):
            PolicyRewardSpec(
                schema_version="pes2ts_policy_reward_spec_v2",
                candidate_bonus=1.0, cost_reference_seconds=100.0,
                hard_failure_penalty=0.25, hard_failure_codes=frozenset(),
                gamma=0.99,
            ).validate()


class TestTransitionAndCost:
    def test_macro_cost_parts_must_sum(self):
        with pytest.raises(ControlContractError, match="sum of its internal"):
            MacroCost(wall_seconds=5.0, host_requests=1,
                      n_optimization_cycles_logged=1,
                      native_wall_seconds=2.0, subdivision_wall_seconds=1.0).validate()

    def test_transition_payload_round_trip(self):
        cost = MacroCost(wall_seconds=1.0, host_requests=1,
                         n_optimization_cycles_logged=2, native_wall_seconds=1.0)
        transition = _make_transition(cost)
        payload = transition.to_payload()
        assert payload["schema_version"] == CONTROL_TRANSITION_SCHEMA_VERSION
        json.dumps(payload)  # serializable
        assert payload["action"]["residual_u"] == [0.5]
        assert payload["decision_start_geometry"] == [[0.0, 0.0, 0.0]]

    def test_transition_rejects_foreign_command_targets(self):
        cost = MacroCost(wall_seconds=1.0, host_requests=1,
                         n_optimization_cycles_logged=1, native_wall_seconds=1.0)
        transition = _make_transition(cost)
        object.__setattr__(transition, "command_targets", (9.9,))
        with pytest.raises(ControlContractError, match="command targets"):
            transition.validate()


def _make_transition(cost: MacroCost) -> object:
    from pes2ts_core.pesn.contracts.control import ControlTransition

    return ControlTransition(
        schema_version=CONTROL_TRANSITION_SCHEMA_VERSION,
        transition_id="t1", environment_id="E_native",
        environment_hash="e" * 64, policy_id="p1", policy_hash="p" * 64,
        reaction_id="RXN_1", decision_index=0, decision_start_frame_index=0,
        action=make_action(), candidates=((0.0,), (0.5,)),
        candidate_sources=(ACTION_SOURCE_RULE_ZERO, ACTION_SOURCE_MODEL_RESIDUAL),
        executed_candidate_index=1, command_targets=(2.05,),
        actual_values=(2.048,), accepted=True, accepted_frame_index=1,
        pending_target=None, child_attempt_refs=("att1",), macro_cost=cost,
        rescue_used=False, refusal_reasons=(), episode_status=TERMINAL_COMPLETE,
        reward=-0.01, reward_events={"candidate_awarded": False, "hard_failure": False},
        bootstrap_mask=False,
        decision_start_geometry=((0.0, 0.0, 0.0),),
        state_after_geometry=((1.0, 0.0, 0.0),),
        next_candidates=(),
        decision_context_vector=(0.5, 0.75, 1.0, 1.0, 0.0, 0.0, 0.0),
    )


class TestCapabilityMapV2:
    def test_v2_map_keeps_formal_caps_disabled(self):
        caps = control_capability_map(residual=True, q_proxy=True)
        validate_control_capability_map(caps)
        assert caps["residual_head"] == "research_only"
        assert caps["q_proxy"] == "research_only"
        for formal in ("p_seed", "c_full", "q_stop", "q_move", "production_control"):
            assert caps[formal] == "disabled"

    def test_v2_map_rejects_enabled_formal_cap(self):
        caps = control_capability_map(residual=True, q_proxy=True)
        caps["q_move"] = "research_only"
        with pytest.raises(Exception, match="disabled"):
            validate_control_capability_map(caps)

    def test_v2_map_rejects_enabled_residual(self):
        caps = control_capability_map(residual=True, q_proxy=True)
        caps["residual_head"] = "enabled"
        with pytest.raises(Exception, match="research_only at most"):
            validate_control_capability_map(caps)

    def test_bundle_manifest_v2_requires_control_fields(self):
        caps = control_capability_map(residual=True, q_proxy=True)
        doc = {
            "schema_version": "pesn_model_bundle_v2",
            "model_id": "m", "model_hash": "a" * 64,
            "feature_spec_sha256": "b" * 64, "label_spec_sha256": "c" * 64,
            "normalization_sha256": "d" * 64, "weights_sha256": "a" * 64,
            "training_run_ids": ["r1"], "capabilities": caps,
            "model_card": "model_card.md", "checksums": "checksums.json",
            "control_spec_sha256": "s" * 64, "environment_id": "E_native",
            "reward_spec_sha256": "w" * 64, "research_only": True,
        }
        validate_control_bundle_manifest(doc)
        del doc["environment_id"]
        with pytest.raises(Exception, match="missing keys"):
            validate_control_bundle_manifest(doc)
