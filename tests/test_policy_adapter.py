"""Policy-adapter tests: candidates, legality, selection, case adaptation."""

from __future__ import annotations

import pytest

from pes2ts_core.generation.planning.policy_adapter import (
    CandidatePolicyConfig,
    ControlCaseAdapter,
    EpsilonExplorePolicy,
    PolicyAdapterError,
    QGreedyPolicy,
    RuleZeroPolicy,
    build_residual_spec,
    generate_candidates,
)
from pes2ts_core.pesn.contracts.control import ControlContext
from tests.test_control_step import build_test_control


class TestResidualSpec:
    def test_derivation_from_control(self):
        control = build_test_control()
        spec = build_residual_spec(control, epsilon_angstrom=0.1)
        assert spec.driver_ids == ("B_1_2",)
        assert spec.min_target[0] == pytest.approx(1.4)
        assert spec.max_target[0] == pytest.approx(2.6)

    def test_non_distance_driver_rejected(self):
        control = build_test_control()
        object.__setattr__(control.drivers[0], "kind", "angle")
        with pytest.raises(PolicyAdapterError, match="distance drivers only"):
            build_residual_spec(control, epsilon_angstrom=0.1)


class TestCandidateGeneration:
    def test_zero_always_first_with_dedup_and_cap(self):
        control = build_test_control()
        spec = build_residual_spec(control, epsilon_angstrom=0.1)
        config = CandidatePolicyConfig(max_candidates=3)
        listing = generate_candidates(spec, control.targets_at(0.25), config=config)
        assert listing.candidates[0] == (0.0,)
        assert listing.sources[0] == "rule_zero"
        assert len(listing.candidates) <= 3
        seen = {tuple(round(v, 9) for v in u) for u in listing.candidates}
        assert len(seen) == len(listing.candidates)

    def test_model_residual_scalings(self):
        control = build_test_control()
        spec = build_residual_spec(control, epsilon_angstrom=0.1)
        config = CandidatePolicyConfig(
            model_scales=(0.5, 1.0), n_single_coord_perturbs=0
        )
        listing = generate_candidates(
            spec, control.targets_at(0.5), config=config, model_u=(0.8,)
        )
        assert (0.4,) in listing.candidates
        assert (0.8,) in listing.candidates
        sources = listing.sources
        assert sources[listing.candidates.index((0.4,))] == "scaled_model_residual"
        assert sources[listing.candidates.index((0.8,))] == "model_residual"

    def test_illegal_model_candidate_dropped_typed(self):
        control = build_test_control()
        spec = build_residual_spec(control, epsilon_angstrom=0.1, allow_sign=1)
        config = CandidatePolicyConfig(model_scales=(1.0,), n_single_coord_perturbs=0)
        listing = generate_candidates(
            spec, control.targets_at(0.5), config=config, model_u=(-0.9,)
        )
        assert listing.candidates == ((0.0,),)
        assert any("sign" in drop for drop in listing.dropped)

    def test_seeded_perturbations_reproducible(self):
        control = build_test_control()
        spec = build_residual_spec(control, epsilon_angstrom=0.1)
        config = CandidatePolicyConfig(n_single_coord_perturbs=2, exploration_seed=7)
        a = generate_candidates(spec, control.targets_at(0.5), config=config)
        b = generate_candidates(spec, control.targets_at(0.5), config=config)
        assert a.candidates == b.candidates
        perturbs = [u for u, s in zip(a.candidates, a.sources)
                    if s == "registered_perturbation"]
        assert len(perturbs) == 2
        for u in perturbs:
            assert abs(u[0]) == pytest.approx(1.0)


class TestSelectionPolicies:
    def _state(self):
        control = build_test_control()
        spec = build_residual_spec(control, epsilon_angstrom=0.1)

        class _State:
            pass

        state = _State()
        state.control = control
        state.spec = spec
        state.target_lambda = 0.5
        return state

    def test_rule_zero_executes_baseline(self):
        policy = RuleZeroPolicy(CandidatePolicyConfig())
        decision = policy.decide(self._state())
        assert decision.executed_candidate_index == 0
        assert decision.candidates[0] == (0.0,)
        assert decision.policy_id == "rule_zero_v1"

    def test_epsilon_explore_reproducible_and_flags_exploration(self):
        config = CandidatePolicyConfig(exploration_probability=0.9, exploration_seed=3)
        policy_a = EpsilonExplorePolicy(config)
        decisions_a = [policy_a.decide(self._state()) for _ in range(6)]
        policy_b = EpsilonExplorePolicy(config)
        decisions_b = [policy_b.decide(self._state()) for _ in range(6)]
        assert [d.executed_candidate_index for d in decisions_a] == \
               [d.executed_candidate_index for d in decisions_b]
        explore_seen = any(
            "bounded_exploration" in decision.candidate_sources
            for decision in decisions_a
        )
        assert explore_seen

    def test_q_greedy_uses_support_and_falls_back(self):
        state = self._state()
        control = state.control
        spec = state.spec

        class _State:
            pass

        real = _State()
        real.control = control
        real.spec = spec
        real.target_lambda = 0.5

        def score_fn(s, candidates):
            values = [None] * len(candidates)
            for index, u in enumerate(candidates):
                if abs(u[0] - 0.4) < 1e-9:
                    values[index] = 5.0
                elif u[0] == 0.0:
                    values[index] = -1.0
            return values

        policy = QGreedyPolicy(
            CandidatePolicyConfig(model_scales=(0.5, 1.0), n_single_coord_perturbs=0),
            score_fn,
            model_residual_fn=lambda s: (0.8,),
        )
        decision = policy.decide(real)
        chosen = decision.candidates[decision.executed_candidate_index]
        assert chosen[0] == pytest.approx(0.4)

        def no_support(s, candidates):
            return [None] * len(candidates)

        fallback = QGreedyPolicy(CandidatePolicyConfig(), no_support)
        decision = fallback.decide(real)
        assert decision.executed_candidate_index == 0


class TestCaseAdapter:
    def test_from_case_and_build_input(self):
        from tests.test_control_step import ORIGIN_GEOMETRY

        control = build_test_control()
        plan = {
            "elements": list(control.elements),
            "atom_map_order": list(control.atom_map_order),
            "active_edits": [{"edit_kind": "broken", "maps": [1, 2]}],
        }
        adapter = ControlCaseAdapter.from_case(
            control=control,
            plan=plan,
            endpoint_geometry_r=ORIGIN_GEOMETRY,
            endpoint_geometry_p=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [4.0, 0.0, 0.0]],
        )
        assert adapter.atomic_numbers == (6, 1, 1)
        spec = build_residual_spec(control, epsilon_angstrom=0.1)
        context = ControlContext(
            nominal_lambda=0.25, next_target_lambda=0.5,
            remaining_wall_fraction=0.9, remaining_request_fraction=0.9,
            recent_rejections=0, rescue_quota_remaining=0,
            candidate_already_awarded=False,
        )
        control_input = adapter.build_input(
            geometry=ORIGIN_GEOMETRY,
            context=context,
            spec=spec,
            base_targets=control.targets_at(0.5),
        )
        assert control_input.input_hash
        assert control_input.base_targets == control.targets_at(0.5)
